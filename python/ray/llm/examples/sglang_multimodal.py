import torch
import asyncio
import pickle
from typing import Dict, Any, Optional, List, Union

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import parse_chat_messages, ChatCompletionMessageParam
from sglang import Engine
from sglang.srt.utils import ImageData

# sglang==0.5.1.post2 doesn't have sglang.srt.parser.jinja_template_utils.process_content_for_template_format
def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
) -> dict:
    """
    Process message content based on detected template format.

    Args:
        msg_dict: Message dictionary with content
        content_format: 'string' or 'openai' (detected via AST analysis)
        image_data: List to append extracted image URLs
        video_data: List to append extracted video URLs
        audio_data: List to append extracted audio URLs
        modalities: List to append modalities

    Returns:
        Processed message dictionary
    """
    if not isinstance(msg_dict.get("content"), list):
        # Already a string or None, no processing needed
        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format == "openai":
        # OpenAI format: preserve structured content list, normalize types
        processed_content_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type == "image_url":
                    image_data.append(
                        ImageData(
                            url=chunk["image_url"]["url"],
                            detail=chunk["image_url"].get("detail", "auto"),
                        )
                    )
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'image' type for template compatibility
                    processed_content_parts.append({"type": "image"})
                elif chunk_type == "video_url":
                    video_data.append(chunk["video_url"]["url"])
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'video' type for template compatibility
                    processed_content_parts.append({"type": "video"})
                elif chunk_type == "audio_url":
                    audio_data.append(chunk["audio_url"]["url"])
                    # Normalize to simple 'audio' type
                    processed_content_parts.append({"type": "audio"})
                else:
                    # Keep other content as-is (text, etc.)
                    processed_content_parts.append(chunk)

        new_msg = {
            k: v for k, v in msg_dict.items() if v is not None and k != "content"
        }
        new_msg["content"] = processed_content_parts
        return new_msg

    elif content_format == "string":
        # String format: flatten to text only (for templates like DeepSeek)
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(chunk["text"])
            # Note: For string format, we ignore images/audio since the template
            # doesn't expect structured content - multimodal placeholders would
            # need to be inserted differently

        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg

    else:
        raise ValueError(f"Invalid content format: {content_format}")

def extract_multimodal_data_for_sglang(
    messages: list[ChatCompletionMessageParam],
    content_format: str = "openai",
) -> tuple[
    Optional[List[Union["ImageData", str]]],  # image_data
    Optional[List[str]],  # audio_data
    Optional[List[str]],  # video_data
]:
    """
    Extract multimodal URLs from OpenAI-format conversation for SGLang.
    
    This function extracts URLs directly from the conversation format,
    similar to how SGLang's process_content_for_template_format works.
    
    Args:
        messages: OpenAI-format conversation messages
        content_format: Content format ("openai" or "string")
    
    Returns:
        Tuple of (image_data, audio_data, video_data) in SGLang-compatible format:
        - image_data: List[ImageData] or List[str] (URLs)
        - audio_data: List[str] (URLs)
        - video_data: List[str] (URLs)
    """
    image_data: List[ImageData] = []
    video_data: List[str] = []
    audio_data: List[str] = []
    modalities: List[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        
        msg_dict = message.copy()
        
        # Ensure content is a list if it's a string (for compatibility)
        if isinstance(msg_dict.get("content"), str):
            msg_dict["content"] = [{"type": "text", "text": msg_dict["content"]}]

        process_content_for_template_format(
            msg_dict,
            content_format,
            image_data,
            video_data,
            audio_data,
            modalities,
        )

    return (
        image_data if image_data else None,
        audio_data if audio_data else None,
        video_data if video_data else None,
    )

async def main():
    model_source = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_config = ModelConfig(
        model=model_source,
    )
    conversation = [
        {"role": "system", "content": "You are a helpful video summarizer."},
        {"role": "user", "content": [
                {"type": "text", "text": f"Describe this video in 3 sentences."},
                {
                    "type": "video_url",
                    "video_url": {"url": "https://content.pexels.com/videos/free-videos.mp4"},
                }
            ]
        },
    ]
    image_data, audio_data, video_data = extract_multimodal_data_for_sglang(conversation)
    print(image_data, audio_data, video_data)

    conversation, mm_data, mm_uuids = parse_chat_messages(
        conversation,
        model_config,
        None,
        content_format="string",
    )
    print(mm_data)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_source)
    prompt = processor.apply_chat_template(conversation, tokenize=False)

    engine_kwargs = {
        "model_path": model_source,
    }

    sglang_engine = Engine(**engine_kwargs)
    stream = await sglang_engine.async_generate(
        prompt=prompt,
        sampling_params={
            "temperature": 0.3,
        },
        image_data=image_data,
        video_data=video_data,
        audio_data=audio_data,
        stream=True,
    )

    async for output in stream:
        print(output)

if __name__ == "__main__":
    asyncio.run(main())