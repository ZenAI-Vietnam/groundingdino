from transformers import AutoTokenizer, BertModel, RobertaModel
import os

def get_tokenlizer(text_encoder_type, use_safetensors=True, local_files_only=True):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )

    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_type,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only
    )
    return tokenizer


def get_pretrained_language_model(text_encoder_type, use_safetensors=True, local_files_only=True):
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(
            text_encoder_type, 
            use_safetensors=use_safetensors,
            local_files_only=local_files_only
        )
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(
            text_encoder_type,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only
        )

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
