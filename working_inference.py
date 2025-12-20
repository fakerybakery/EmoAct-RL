# server_tts_final_robust_gpu.py
import os
# --- CRITICAL: FORCE STABLE ENGINE ---
os.environ["VLLM_USE_V1"] = "0" 
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import uuid
import random
import base64
import logging
import multiprocessing as mp
import subprocess
import shutil
import numpy as np
import wave
import traceback
import threading
import inspect
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler

# --- CONFIGURATION ---
MODEL_DIR = "/dev/shm/vocalino_monster" #"/home/deployer/laion/Vocalino_0.11_alpha"
OFFLINE_ROOT = "/home/deployer/laion/offline_orpheus_bundle"
FFMPEG_EXE = os.path.join(OFFLINE_ROOT, "bin", "ffmpeg")
HF_CACHE = os.path.join(OFFLINE_ROOT, "hf_cache")
OUTPUT_DIR = "server_output"
REF_UPLOAD_DIR = "/dev/shm/ref_uploads"
LOG_FILE = "server_tts_batching.log"
TEMP_DIR = "/dev/shm/tts_processing_tmp"

# --- BATCHING CONFIGURATION ---
GATHER_INTERVAL_SECONDS = 0.1
INITIAL_MAX_BATCH_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Environment
os.environ["HF_HOME"] = HF_CACHE
os.environ["VLLM_CACHE_DIR"] = HF_CACHE

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=20*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- SCHEMAS ---
class Range(BaseModel): min: float; max: float
class GenerationTask(BaseModel): prompt_str: str; text_to_say: str; ref_audio_path: Optional[str] = None
class BatchRequest(BaseModel):
    tasks: List[GenerationTask]
    num_samples_per_task: int = 1
    temperature: Union[float, Range] = 0.7
    repetition_penalty: Union[float, Range] = 1.2
    presence_penalty: Union[float, Range] = 0.0
    max_new_tokens: int = 4000
    top_p: float = 0.9

# --- GPU WORKER CLASS ---
class GpuWorker:
    def __init__(self, gpu_id: int, input_queue, result_queue):
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.worker_name = f"Worker-{self.gpu_id}"
        self.max_safe_batch_size = INITIAL_MAX_BATCH_SIZE

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        try:
            import torch
            import vllm
            from vllm import SamplingParams
            from vllm.engine.arg_utils import EngineArgs
            from vllm.engine.llm_engine import LLMEngine
            from transformers import AutoTokenizer
            from snac import SNAC
            import glob

            try:
                from vllm.inputs.data import TokensPrompt
                HAS_TOKENS_PROMPT = True
            except ImportError:
                HAS_TOKENS_PROMPT = False

            logger.info(f"{self.worker_name}: Using vLLM version {vllm.__version__}")
            device = torch.device("cuda:0")
            
            logger.info(f"{self.worker_name}: Initializing models...")
            
            # --- MEMORY CONFIG: 30% for LLM, 70% free for SNAC/Other ---
            engine_args = EngineArgs(model=MODEL_DIR, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.30, enforce_eager=True)
            engine = LLMEngine.from_engine_args(engine_args)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            
            sig = inspect.signature(engine.add_request)
            params_names = list(sig.parameters.keys())
            
            snac_path_glob = glob.glob(os.path.join(HF_CACHE, "hub", "models--hubertsiuzdak--snac_24khz", "snapshots", "*"))
            if not snac_path_glob: raise RuntimeError("SNAC model not found.")
            
            # --- LOAD SNAC ON GPU ---
            snac_model = SNAC.from_pretrained(snac_path_glob[0]).to(device).eval()
            logger.info(f"{self.worker_name}: SNAC model loaded to GPU {device}.")
            
            logger.info(f"{self.worker_name}: Ready.")

            start_of_speech, end_of_speech = 128257, 128258
            start_of_human, end_of_human = 128259, 128260
            start_of_ai = 128261
            TOKEN_OFFSET_BASE = 128266
            LAYER_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]
            
            # Separate CUDA stream for SNAC
            snac_stream = torch.cuda.Stream(device=device)

            while True:
                try:
                    batch = self.input_queue.get()
                    if not batch: continue
                    batch_size = len(batch)
                    
                    if batch_size > self.max_safe_batch_size:
                        logger.warning(f"{self.worker_name}: Splitting batch {batch_size}")
                        mid = batch_size // 2
                        self.input_queue.put(batch[:mid]); self.input_queue.put(batch[mid:])
                        continue

                    logger.info(f"{self.worker_name}: Starting batch of {batch_size}")
                    t_gen_start = time.perf_counter()
                    batch_request_ids = {}
                    
                    for i, job in enumerate(batch):
                        ref_tokens = []
                        if job['ref_audio'] and os.path.exists(job['ref_audio']):
                            tmp_in = os.path.join(TEMP_DIR, f"ref_in_{job['req_id']}_{self.gpu_id}.wav")
                            try:
                                subprocess.run([FFMPEG_EXE, "-y", "-v", "error", "-i", job['ref_audio'], "-ar", "24000", "-ac", "1", "-f", "wav", tmp_in], check=True, timeout=10)
                                with wave.open(tmp_in, 'rb') as wf:
                                    raw = wf.readframes(wf.getnframes()); audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                                
                                # --- SANITIZATION 1: Check Input Audio ---
                                if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                                    logger.error(f"{self.worker_name}: Input audio contains NaNs or Infs! Skipping ref audio.")
                                else:
                                    # --- FIX: Move Input Tensor to GPU ---
                                    with torch.cuda.stream(snac_stream):
                                        tens = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(1).to(device)
                                        with torch.inference_mode(): codes = snac_model.encode(tens)
                                        c0, c1, c2 = codes[0].cpu().numpy()[0], codes[1].cpu().numpy()[0], codes[2].cpu().numpy()[0]
                                    
                                    torch.cuda.current_stream().wait_stream(snac_stream)

                                    for j in range(len(c0)):
                                        ref_tokens.extend([
                                            c0[j] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[0], c1[2*j] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[1],
                                            c2[4*j] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[2], c2[4*j+1] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[3],
                                            c1[2*j+1] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[4], c2[4*j+2] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[5],
                                            c2[4*j+3] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[6]
                                        ])
                            except Exception as e:
                                logger.error(f"{self.worker_name}: Ref audio processing failed: {e}")
                            finally:
                                if os.path.exists(tmp_in): os.remove(tmp_in)
                        
                        input_ids = [start_of_human]
                        if ref_tokens: input_ids += tokenizer.encode("Reference audio: ", add_special_tokens=False) + [start_of_speech] + ref_tokens + [end_of_speech]
                        input_ids += tokenizer.encode("Text: " + job['prompt'], add_special_tokens=False)
                        input_ids += [128009, end_of_human, start_of_ai, start_of_speech]
                        input_ids = [int(x) for x in input_ids]
                        
                        params = job['params']
                        stop_ids = [end_of_speech]
                        if tokenizer.eos_token_id is not None: stop_ids.append(tokenizer.eos_token_id)
                        
                        sampling_params = SamplingParams(
                            temperature=params['temp'], top_p=params['top_p'], repetition_penalty=params['rep_pen'],
                            presence_penalty=params['pres_pen'], max_tokens=params['max_new_tokens'], stop_token_ids=stop_ids,
                        )
                        
                        request_id = str(uuid.uuid4())
                        batch_request_ids[request_id] = i
                        
                        # Add Request
                        if HAS_TOKENS_PROMPT and 'params' in params_names and 'prompt' in params_names:
                            prompt_obj = TokensPrompt(prompt_token_ids=input_ids)
                            engine.add_request(request_id, prompt=prompt_obj, params=sampling_params)
                        elif 'inputs' in params_names:
                            prompt_dict = {"prompt_token_ids": input_ids}
                            engine.add_request(request_id, inputs=prompt_dict, sampling_params=sampling_params)
                        elif 'prompt' in params_names and 'sampling_params' in params_names:
                            engine.add_request(request_id, prompt=None, sampling_params=sampling_params, prompt_token_ids=input_ids)
                        else:
                            kw_args = {'prompt_token_ids': input_ids}
                            sp_key = 'params' if 'params' in params_names else 'sampling_params'
                            kw_args[sp_key] = sampling_params
                            engine.add_request(request_id, prompt=None, **kw_args)

                    outputs = [None] * batch_size
                    while True:
                        request_outputs = engine.step()
                        for output in request_outputs:
                            if output.finished:
                                original_index = batch_request_ids.pop(output.request_id, None)
                                if original_index is not None:
                                    outputs[original_index] = output
                        if not batch_request_ids: break
                    
                    t_gen_end = time.perf_counter()
                    logger.info(f"{self.worker_name}: VLLM gen took {t_gen_end - t_gen_start:.4f}s")
                    
                    # --- DECODE AND SAVE (ON GPU) ---
                    for idx, (job, output) in enumerate(zip(batch, outputs)):
                        if output is None:
                            logger.error(f"{self.worker_name}: Job {idx} returned None.")
                            self.result_queue.put({'status': 'error', 'req_id': job['req_id'], 'error': 'vLLM output None'})
                            continue
                        
                        logger.info(f"{self.worker_name}: Req {idx} finish reason: {output.outputs[0].finish_reason}")

                        try:
                            gen_ids = output.outputs[0].token_ids
                            audio_tokens = [t for t in gen_ids if t >= TOKEN_OFFSET_BASE]
                            if not audio_tokens: raise ValueError("No audio tokens generated")
                            
                            c0, c1, c2 = [], [], []
                            ptr = 0
                            while ptr + 7 <= len(audio_tokens):
                                chunk = audio_tokens[ptr:ptr+7]
                                c0.append(chunk[0] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[0])
                                c1.extend([chunk[1] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[1], chunk[4] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[4]])
                                c2.extend([chunk[2] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[2], chunk[3] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[3], chunk[5] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[5], chunk[6] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[6]])
                                ptr += 7
                            if not c0: raise ValueError("SNAC parse error")
                            
                            # --- SANITIZATION 2: Check Codebook Indices ---
                            # Prevent "Device-side assert triggered" (Index Out of Bounds)
                            # SNAC codebooks are size 4096.
                            def sanitize_indices(lst):
                                return [max(0, min(4095, x)) for x in lst]
                            
                            c0 = sanitize_indices(c0)
                            c1 = sanitize_indices(c1)
                            c2 = sanitize_indices(c2)

                            # --- GPU DECODING ---
                            with torch.cuda.stream(snac_stream):
                                z0 = torch.tensor(c0, device=device).unsqueeze(0)
                                z1 = torch.tensor(c1, device=device).unsqueeze(0)
                                z2 = torch.tensor(c2, device=device).unsqueeze(0)
                                with torch.inference_mode():
                                    out_wav = snac_model.decode([z0, z1, z2]).squeeze()
                                    out_wav_cpu = out_wav.cpu().numpy()
                            
                            torch.cuda.current_stream().wait_stream(snac_stream)
                            
                            int_wav = (np.clip(out_wav_cpu, -1.0, 1.0) * 32767).astype(np.int16)
                            tmp_wav = os.path.join(OUTPUT_DIR, f"temp_{job['req_id']}_{job['params']['global_idx']}.wav")
                            with wave.open(tmp_wav, 'wb') as wf: wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(int_wav.tobytes())
                            
                            final_filename = f"gen_{job['req_id']}_{job['params']['global_idx']}.mp3"
                            local_path = os.path.abspath(os.path.join(OUTPUT_DIR, final_filename))
                            
                            subprocess.run([FFMPEG_EXE, "-y", "-v", "error", "-i", tmp_wav, "-ac", "1", "-b:a", "96k", "-f", "mp3", local_path], check=True, timeout=5)
                            
                            if os.path.exists(tmp_wav): os.remove(tmp_wav)
                            self.result_queue.put({'status': 'success', 'req_id': job['req_id'], 'local_path': local_path, 'duration': len(int_wav)/24000, 'params': job['params']})
                        
                        except Exception as e:
                            logger.error(f"{self.worker_name}: Error in post-processing: {e}")
                            traceback.print_exc()
                            self.result_queue.put({'status': 'error', 'req_id': job['req_id'], 'error': str(e)})
                    
                    logger.info(f"{self.worker_name}: Batch complete.")

                except Exception as e:
                    logger.error(f"{self.worker_name}: Unhandled error in worker loop: {e}")
                    traceback.print_exc()

        except Exception as e:
            logger.critical(f"{self.worker_name} CRITICAL CRASH: {e}")
            traceback.print_exc()

# --- MAIN BLOCK & DISPATCHER ---
ctx = mp.get_context('spawn')
request_pool, request_pool_lock = [], threading.Lock()
worker_queues, q_out, results_store = [], ctx.Queue(), {}
app = FastAPI()

def dispatcher_loop(worker_queues):
    num_workers = len(worker_queues)
    logger.info(f"[Dispatcher]: Starting up. Will dispatch to {num_workers} workers every {GATHER_INTERVAL_SECONDS}s.")
    while True:
        time.sleep(GATHER_INTERVAL_SECONDS)
        jobs_to_dispatch = []
        with request_pool_lock:
            if request_pool: jobs_to_dispatch.extend(request_pool); request_pool.clear()
        if not jobs_to_dispatch: continue
        logger.info(f"[Dispatcher]: Awakened. Found {len(jobs_to_dispatch)} new requests to distribute.")
        job_chunks = np.array_split(jobs_to_dispatch, num_workers)
        for i, chunk in enumerate(job_chunks):
            if len(chunk) > 0:
                batch = list(chunk)
                worker_queues[i].put(batch)
                logger.info(f"[Dispatcher]: Sent batch of size {len(batch)} to {getattr(worker_queues[i], 'name', f'Worker-{i}')}.")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
@app.post("/upload_ref")
async def upload_ref(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".wav"; filename = f"ref_{uuid.uuid4()}{ext}"
        file_path = os.path.join(REF_UPLOAD_DIR, filename)
        with open(file_path, "wb") as f: shutil.copyfileobj(file.file, f)
        return {"status": "success", "local_path": os.path.abspath(file_path)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

def collector():
    while True:
        try:
            res = q_out.get(); rid = res.get('req_id')
            if rid and rid in results_store: results_store[rid].append(res)
        except: time.sleep(0.1)

@app.post("/generate_batch")
def generate_batch(request: BatchRequest):
    req_id = str(uuid.uuid4()); results_store[req_id] = []
    total_jobs, new_jobs = 0, []
    for task in request.tasks:
        for i in range(request.num_samples_per_task):
            def get_val(v): return random.uniform(v.min, v.max) if isinstance(v, Range) else v
            job = {
                'req_id': req_id, 'prompt': f"{task.prompt_str}. {task.text_to_say}", 'ref_audio': task.ref_audio_path,
                'params': {'temp': get_val(request.temperature), 'rep_pen': get_val(request.repetition_penalty),
                           'pres_pen': get_val(request.presence_penalty), 'max_new_tokens': request.max_new_tokens,
                           'top_p': get_val(request.top_p), 'global_idx': total_jobs}
            }
            new_jobs.append(job); total_jobs += 1
    
    with request_pool_lock: request_pool.extend(new_jobs)
    logger.info(f"Added {total_jobs} jobs to the request pool for request ID {req_id}")

    start_time = time.time()
    while len(results_store.get(req_id, [])) < total_jobs:
        time.sleep(0.1)
        if time.time() - start_time > 600:
            if req_id in results_store: del results_store[req_id]
            return JSONResponse({"error": "Request timed out"}, status_code=504)

    final_output = []
    raw_results = results_store.pop(req_id, [])
    for res in raw_results:
        if res['status'] == 'error': continue
        try:
            with open(res['local_path'], "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
            final_output.append({"sample_id": str(uuid.uuid4()), "audio_base64": b64, "duration_sec": res['duration'],
                                 "local_path": res['local_path'], "params": res['params']})
        except Exception as e: logger.error(f"Error reading result file {res.get('local_path')}: {e}")
    return final_output

if __name__ == "__main__":
    import torch
    try: n_gpus = torch.cuda.device_count()
    except Exception as e: logger.critical(f"Could not detect CUDA devices: {e}"); n_gpus = 0
    if n_gpus == 0: logger.critical("No GPUs found. Exiting."); sys.exit(1)
    
    logger.info(f"Found {n_gpus} CUDA-enabled GPUs.")
    processes = []
    for i in range(n_gpus):
        q = ctx.Queue(); setattr(q, 'name', f'Worker-{i}')
        worker_queues.append(q)
        worker = GpuWorker(gpu_id=i, input_queue=q, result_queue=q_out)
        p = ctx.Process(target=worker.run, name=f"Worker-{i}"); p.start()
        processes.append(p)

    collector_thread = threading.Thread(target=collector, daemon=True); collector_thread.start()
    dispatcher_thread = threading.Thread(target=dispatcher_loop, args=(worker_queues,), daemon=True); dispatcher_thread.start()
    import uvicorn
    logger.info(f"Starting FastAPI server with dispatcher and {n_gpus} GPU workers.")
    uvicorn.run(app, host="0.0.0.0", port=8010)
