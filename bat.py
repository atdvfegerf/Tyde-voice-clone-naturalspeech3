import wget
urls = [
    "https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder.bin?download=true",
    "https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder_v2.bin?download=true",
    "https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder.bin?download=true"
    "https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder_v2.bin?download=true"
    "https://huggingface.co/spaces/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_redecoder.bin?download=true"
]
directory = "/content/Tyde-voice-clone-naturalspeech3"
for url in urls:
    wget.download(url, out=directory)
