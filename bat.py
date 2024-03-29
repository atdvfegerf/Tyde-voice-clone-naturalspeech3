!pip install wget

import wget
import os

# Lista di tuple contenenti l'URL del file e il nome desiderato del file
file_info = [
    ("https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder.bat?download=true", "ns3_facodec_decoder.bat"),
    ("https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder_v2.bat?download=true", "ns3_facodec_decoder_v2.bat"),
    ("https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder.bat?download=true", "ns3_facodec_encoder.bat"),
    ("https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder_v2.bat?download=true", "ns3_facodec_encoder_v2.bat"),
    ("https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_redecoder.bat?download=true", "ns3_facodec_redecoder.bat")
]

# Directory di destinazione
directory = "/content/Tyde-voice-clone-naturalspeech3 "  # Cambia la directory in quella che desideri

# Scarica ciascun file nella directory specificata
for url, file_name in file_info:
    # Rinomina il file locale se esiste già
    local_file_path = os.path.join(directory, file_name)
    if os.path.exists(local_file_path):
        base_name, ext = os.path.splitext(file_name)
        file_name = f"{base_name}_downloaded{ext}"
    # Scarica il file con il nome desiderato nella directory specificata
    wget.download(url, out=directory, bar=None, out_filename=file_name)
