import glob
import os
import json
import numpy as np
import torch
import argparse
import torchaudio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional
from muq import MuQ
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm


class Synthesizer(object):

    def __init__(self,
                 checkpoint_path,
                 input_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_cpu: bool = False):
        self.checkpoint_path = checkpoint_path
        self.input_path = input_path
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device('cuda') if (torch.cuda.is_available() and (not use_cpu)) else torch.device('cpu')
        self.result_dict: dict[str, dict[str, float]] = {}
        self._is_setup = False

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        return path.startswith("http://") or path.startswith("https://")

    @staticmethod
    def _load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
        """Load local path or remote URL with torchaudio only, returning [1, T]."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    
    @torch.no_grad()
    def setup(self):
        train_config = OmegaConf.load(os.path.join(os.path.dirname(self.checkpoint_path), '../config.yaml'))
        model = instantiate(train_config.generator).to(self.device).eval()
        state_dict = load_file(self.checkpoint_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = self.muq.to(self.device).eval()
        self._is_setup = True

    def _collect_input_files(self, input_path: str) -> list[str]:
        if os.path.isfile(input_path):
            if input_path.endswith(('.wav', '.mp3', '.npy')) or self._is_remote_path(input_path):
                return [input_path]
            with open(input_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        if os.path.isdir(input_path):
            files = glob.glob(os.path.join(input_path, '*'))
            return [
                file for file in files
                if file.lower().endswith(('.wav', '.mp3', '.npy'))
            ]
        if self._is_remote_path(input_path):
            return [input_path]
        raise ValueError(f"input_path {input_path} is not a file, directory, or URL")

    @staticmethod
    def _file_id_from_path(input_path: str) -> str:
        file_path = input_path.split('?')[0]
        return os.path.basename(file_path).split('.')[0]

    @torch.no_grad()
    def handle(self, input_path: str, preloaded_audio: Optional[torch.Tensor] = None) -> dict[str, float] | None:
        fid = self._file_id_from_path(input_path)
        if input_path.endswith('.npy'):
            input = np.load(input_path)

            if len(input.shape) == 3 and input.shape[0] != 1:
                print('ssl_shape error', input_path)
                return None
            if np.isnan(input).any():
                print('ssl nan', input_path)
                return None

            input = torch.from_numpy(input).to(self.device)
            if len(input.shape) == 2:
                input = input.unsqueeze(0)

        if input_path.endswith(('.wav', '.mp3')) or self._is_remote_path(input_path):
            audio = preloaded_audio if preloaded_audio is not None else self._load_audio(input_path, target_sr=24000)
            audio = audio.to(self.device)
            output = self.muq(audio, output_hidden_states=True)
            input = output["hidden_states"][6]

        values = {}
        scores_g = self.model(input).squeeze(0)
        values['Coherence'] = round(scores_g[0].item(), 4)
        values['Musicality'] = round(scores_g[1].item(), 4)
        values['Memorability'] = round(scores_g[2].item(), 4)
        values['Clarity'] = round(scores_g[3].item(), 4)
        values['Naturalness'] = round(scores_g[4].item(), 4)

        self.result_dict[fid] = values
        return values

    def _is_audio_path(self, input_path: str) -> bool:
        return input_path.endswith(('.wav', '.mp3')) or self._is_remote_path(input_path)

    @torch.no_grad()
    def predict(
        self,
        input_path: str,
        reset_results: bool = True,
        show_progress: bool = True,
        prefetch_workers: int = 0,
        prefetch_buffer: int = 8,
    ) -> dict[str, dict[str, float]]:
        if not self._is_setup:
            self.setup()
        if reset_results:
            self.result_dict = {}

        input_files = self._collect_input_files(input_path)
        if prefetch_workers <= 0:
            iterator = tqdm(input_files) if show_progress else input_files
            for path in iterator:
                self.handle(path)
            return self.result_dict

        prefetch_buffer = max(1, prefetch_buffer)
        in_flight: dict[int, Future | None] = {}
        total = len(input_files)
        next_submit = 0

        def submit_until_full(pool: ThreadPoolExecutor) -> None:
            nonlocal next_submit
            while next_submit < total and len(in_flight) < prefetch_buffer:
                path = input_files[next_submit]
                if self._is_audio_path(path):
                    in_flight[next_submit] = pool.submit(self._load_audio, path, 24000)
                else:
                    in_flight[next_submit] = None
                next_submit += 1

        with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
            submit_until_full(pool)
            progress = tqdm(total=total) if show_progress else None
            for idx, path in enumerate(input_files):
                submit_until_full(pool)
                future = in_flight.pop(idx, None)
                preloaded_audio = future.result() if isinstance(future, Future) else None
                self.handle(path, preloaded_audio=preloaded_audio)
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()
        return self.result_dict

    def to_rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for file_id, metrics in self.result_dict.items():
            rows.append({"file_id": file_id, **metrics})
        return rows

    def save_results(self, output_dir: str, file_name: str = "result.json") -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w") as f:
            json.dump(self.result_dict, f, indent=4, ensure_ascii=False)
        return output_path

    @torch.no_grad()
    def synthesis(self):
        if self.input_path is None:
            raise ValueError("input_path must be set for synthesis(). Use predict(input_path=...) for library usage.")
        results = self.predict(self.input_path, reset_results=True, show_progress=True)
        if self.output_dir:
            self.save_results(self.output_dir)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Input audio: local file path, signed URL, text file listing paths/URLs, or directory of audio files."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Optional output directory for result.json."
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU mode even if a GPU is available.",
    )
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=0,
        help="Number of CPU threads for audio decode prefetch. 0 disables prefetch.",
    )
    parser.add_argument(
        "--prefetch_buffer",
        type=int,
        default=8,
        help="Max number of prefetched items kept in RAM.",
    )

    args = parser.parse_args()

    ckpt_path = "ckpt/model.safetensors"

    synthesizer = Synthesizer(checkpoint_path=ckpt_path,
                              input_path=args.input_path,
                              output_dir=args.output_dir,
                              use_cpu=args.use_cpu)

    result_dict = synthesizer.predict(
        input_path=args.input_path,
        reset_results=True,
        show_progress=True,
        prefetch_workers=args.prefetch_workers,
        prefetch_buffer=args.prefetch_buffer,
    )
    if args.output_dir:
        synthesizer.save_results(args.output_dir)
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))