import torch
import torchaudio
import argparse
import os

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def process_audio(input_path, output_denoised, output_enhanced, solver, nfe, tau, denoising):
    if not os.path.isfile(input_path):
        print(f"Input file {input_path} does not exist.")
        return

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(input_path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    # Save the output audio files
    torchaudio.save(output_denoised, wav1.unsqueeze(0), new_sr)
    torchaudio.save(output_enhanced, wav2.unsqueeze(0), new_sr)
    print(f"Denoised audio saved to {output_denoised}")
    print(f"Enhanced audio saved to {output_enhanced}")


def main():
    parser = argparse.ArgumentParser(description="Enhance audio using Resemble Enhance")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output-denoised", default="output_denoised.wav", help="Path to save denoised audio")
    parser.add_argument("--output-enhanced", default="output_enhanced.wav", help="Path to save enhanced audio")
    parser.add_argument("--solver", default="Midpoint", choices=["Midpoint", "RK4", "Euler"], help="ODE solver")
    parser.add_argument("--nfe", type=int, default=64, help="Number of function evaluations")
    parser.add_argument("--tau", type=float, default=0.5, help="Prior temperature (0-1)")
    parser.add_argument("--denoising", action="store_true", help="Denoise before enhancement")

    args = parser.parse_args()

    process_audio(
        args.input,
        args.output_denoised,
        args.output_enhanced,
        args.solver,
        args.nfe,
        args.tau,
        args.denoising
    )

if __name__ == "__main__":
    main()
