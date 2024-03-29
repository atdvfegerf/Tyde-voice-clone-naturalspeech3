import spaces
import torch
import librosa
import soundfile as sf
import gradio as gr
import torchaudio
import os
from huggingface_hub import hf_hub_download
import numpy as np

from Amphion.models.ns3_codec import (
    FACodecEncoder,
    FACodecDecoder,
    FACodecRedecoder,
    FACodecEncoderV2,
    FACodecDecoderV2,
)

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

fa_redecoder = FACodecRedecoder()

fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder_v2 = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)


# encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
# decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
# redecoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_redecoder.bin")
# encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
# decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

encoder_ckpt = "ns3_facodec_encoder.bin"
decoder_ckpt = "ns3_facodec_decoder.bin"
redecoder_ckpt = "ns3_facodec_redecoder.bin"
encoder_v2_ckpt = "ns3_facodec_encoder_v2.bin"
decoder_v2_ckpt = "ns3_facodec_decoder_v2.bin"

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))
fa_redecoder.load_state_dict(torch.load(redecoder_ckpt))
fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fa_encoder = fa_encoder.to(device)
fa_decoder = fa_decoder.to(device)
fa_redecoder = fa_redecoder.to(device)
fa_encoder_v2 = fa_encoder_v2.to(device)
fa_decoder_v2 = fa_decoder_v2.to(device)
fa_encoder.eval()
fa_decoder.eval()
fa_redecoder.eval()
fa_encoder_v2.eval()
fa_decoder_v2.eval()

@spaces.GPU
def codec_inference(speech_path):

    with torch.no_grad():

        wav, sr = librosa.load(speech_path, sr=16000)
        wav = torch.tensor(wav).to(device).unsqueeze(0).unsqueeze(0)

        enc_out = fa_encoder(wav)
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(
            enc_out, eval_vq=False, vq=True
        )
        recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)

    os.makedirs("temp", exist_ok=True)
    result_path = "temp/result.wav"
    sf.write(result_path, recon_wav[0, 0].cpu().numpy(), 16000)

    return result_path

@spaces.GPU
def codec_voice_conversion(speech_path_a, speech_path_b):

    with torch.no_grad():

        wav_a, sr = librosa.load(speech_path_a, sr=16000)
        wav_a = torch.tensor(wav_a).to(device).unsqueeze(0).unsqueeze(0)
        wav_b, sr = librosa.load(speech_path_b, sr=16000)
        wav_b = torch.tensor(wav_b).to(device).unsqueeze(0).unsqueeze(0)

        enc_out_a = fa_encoder(wav_a)
        enc_out_b = fa_encoder(wav_b)

        vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder(
            enc_out_a, eval_vq=False, vq=True
        )
        vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder(
            enc_out_b, eval_vq=False, vq=True
        )

        recon_wav_a = fa_decoder.inference(vq_post_emb_a, spk_embs_a)
        recon_wav_b = fa_decoder.inference(vq_post_emb_b, spk_embs_b)

        vq_post_emb_a_to_b = fa_redecoder.vq2emb(
            vq_id_a, spk_embs_b, use_residual=False
        )
        recon_wav_a_to_b = fa_redecoder.inference(vq_post_emb_a_to_b, spk_embs_b)

    os.makedirs("temp", exist_ok=True)
    recon_a_result_path = "temp/result_a.wav"
    recon_b_result_path = "temp/result_b.wav"
    vc_result_path = "temp/result_vc.wav"
    sf.write(vc_result_path, recon_wav_a_to_b[0, 0].cpu().numpy(), 16000)
    sf.write(recon_a_result_path, recon_wav_a[0, 0].cpu().numpy(), 16000)
    sf.write(recon_b_result_path, recon_wav_b[0, 0].cpu().numpy(), 16000)

    return recon_a_result_path, recon_b_result_path, vc_result_path

@spaces.GPU
def codec_voice_conversion_v2(speech_path_a, speech_path_b):

    with torch.no_grad():

        wav_a, sr = librosa.load(speech_path_a, sr=16000)
        wav_a = np.pad(wav_a, (0, 200 - len(wav_a) % 200))
        wav_a = torch.tensor(wav_a).to(device).unsqueeze(0).unsqueeze(0)
        wav_b, sr = librosa.load(speech_path_b, sr=16000)
        wav_b = np.pad(wav_b, (0, 200 - len(wav_b) % 200))
        wav_b = torch.tensor(wav_b).to(device).unsqueeze(0).unsqueeze(0)

        enc_out_a = fa_encoder_v2(wav_a)
        prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
        enc_out_b = fa_encoder_v2(wav_b)
        prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

        vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
            enc_out_a, prosody_a, eval_vq=False, vq=True
        )
        vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
            enc_out_b, prosody_b, eval_vq=False, vq=True
        )

        recon_wav_a = fa_decoder_v2.inference(vq_post_emb_a, spk_embs_a)
        recon_wav_b = fa_decoder_v2.inference(vq_post_emb_b, spk_embs_b)

        vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
        recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)

    os.makedirs("temp", exist_ok=True)
    recon_a_result_path = "temp/result_a.wav"
    recon_b_result_path = "temp/result_b.wav"
    vc_result_path = "temp/result_vc.wav"
    sf.write(vc_result_path, recon_wav_a_to_b[0, 0].cpu().numpy(), 16000)
    sf.write(recon_a_result_path, recon_wav_a[0, 0].cpu().numpy(), 16000)
    sf.write(recon_b_result_path, recon_wav_b[0, 0].cpu().numpy(), 16000)

    return recon_a_result_path, recon_b_result_path, vc_result_path

demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload the speech file",
        type="filepath",
    ),
]

demo_outputs = [
    gr.Audio(label="Speech reconstructed"),
]

vc_demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload the source speech file",
        type="filepath",
    ),
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload the reference speech file",
        type="filepath",
    ),
]

vc_demo_outputs = [
    gr.Audio(label="Source speech reconstructed"),
    gr.Audio(label="Reference speech reconstructed"),
    gr.Audio(label="Voice conversion result"),
]

with gr.Blocks() as demo:
    gr.Interface(
        fn=codec_inference,
        inputs=demo_inputs,
        outputs=demo_outputs,
        title="FACodec for NaturalSpeech 3",
        description="""
        ## FACodec: Speech Codec with Attribute Factorization used for NaturalSpeech 3

        [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2403.03100.pdf)

        [![demo](https://img.shields.io/badge/FACodec-Demo-red)](https://speechresearch.github.io/naturalspeech3/)

        [![model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-pink)](https://huggingface.co/amphion/naturalspeech3_facodec)

        ## Overview

        FACodec is a core component of the advanced text-to-speech (TTS) model NaturalSpeech 3. FACodec converts complex speech waveform into disentangled subspaces representing speech attributes of content, prosody, timbre, and acoustic details and reconstruct high-quality speech waveform from these attributes. FACodec decomposes complex speech into subspaces representing different attributes, thus simplifying the modeling of speech representation.

        Research can use FACodec to develop different modes of TTS models, such as non-autoregressive based discrete diffusion (NaturalSpeech 3) or autoregressive models (like VALL-E).
        """,
    )

    gr.Examples(
        examples=[
            [
                "default/ref/ref.wav"
            ],
        ],
        inputs=demo_inputs,
    )
        
    gr.Interface(
        fn=codec_voice_conversion_v2,
        inputs=vc_demo_inputs,
        outputs=vc_demo_outputs,
        title="FACodec Voice Conversion",
        description="""
        FACodec can achieve zero-shot voice conversion. 
        """,
    )

    gr.Examples(
        examples=[
            [
                "default/source/source.wav",
                "default/ref/ref.wav",
            ],
        ],
        inputs=vc_demo_inputs,
    )

    demo.queue()
    demo.launch(share=True)

