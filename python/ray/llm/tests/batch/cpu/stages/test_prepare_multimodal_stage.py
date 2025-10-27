import pytest
import io
import requests
import PIL.Image

from ray.llm._internal.batch.stages.prepare_multimodal_stage import (
    PrepareMultimodalUDF,
)


@pytest.mark.asyncio
async def test_prepare_multimodal_udf_image_url():
    udf = PrepareMultimodalUDF(
        data_column="__data", 
        expected_input_keys=["messages"],
        model="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    batch = {
        "__data": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Describe this image in 10 words."},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}
                            }
                        ]
                    }
                ]
            }
        ]
    }

    results = []
    async for result in udf(batch):
        results.append(result["__data"][0])

    assert len(results) == 1
    assert "multimodal_data" in results[0]
    assert "messages" in results[0]


@pytest.mark.asyncio
async def test_prepare_multimodal_udf_pil_image():
    image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
    response = requests.get(image_url)
    image_pil = PIL.Image.open(io.BytesIO(response.content))
    
    udf = PrepareMultimodalUDF(
        data_column="__data", 
        expected_input_keys=["messages"],
        model="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    batch = {
        "__data": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Describe this image in 10 words."},
                            {
                                "type": "image_pil",
                                "image_pil": image_pil,
                            }
                        ]
                    }
                ]
            }
        ]
    }

    results = []
    async for result in udf(batch):
        results.append(result["__data"][0])

    assert len(results) == 1
    assert "multimodal_data" in results[0]
    assert "messages" in results[0]


# @pytest.mark.asyncio
# async def test_prepare_multimodal_udf_video_url():
#     udf = PrepareMultimodalUDF(
#         data_column="__data", 
#         expected_input_keys=["messages"],
#         model="Qwen/Qwen2.5-VL-3B-Instruct"
#     )

#     # Test batch with video URL (using a real video)
#     batch = {
#         "__data": [
#             {
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user", 
#                         "content": [
#                             {"type": "text", "text": "Describe this video in 10 words."},
#                             {
#                                 "type": "video_url",
#                                 "video_url": {"url": "https://content.pexels.com/videos/free-videos.mp4"}
#                             }
#                         ]
#                     }
#                 ]
#             }
#         ]
#     }

#     results = []
#     async for result in udf(batch):
#         results.append(result["__data"][0])

#     assert len(results) == 1
#     assert "multimodal_data" in results[0]
#     assert "messages" in results[0]


# @pytest.mark.asyncio
# async def test_prepare_multimodal_udf_multiple_items():
#     udf = PrepareMultimodalUDF(
#         data_column="__data", 
#         expected_input_keys=["messages"],
#         model="Qwen/Qwen2.5-VL-3B-Instruct"
#     )

#     # Test batch with multiple items (using real URLs)
#     batch = {
#         "__data": [
#             {
#                 "messages": [
#                     {"role": "user", "content": [
#                         {"type": "text", "text": "Describe this image."},
#                         {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}}
#                     ]}
#                 ]
#             },
#             {
#                 "messages": [
#                     {"role": "user", "content": [
#                         {"type": "text", "text": "What's in this video?"},
#                         {"type": "video_url", "video_url": {"url": "https://content.pexels.com/videos/free-videos.mp4"}}
#                     ]}
#                 ]
#             }
#         ]
#     }

#     results = []
#     async for result in udf(batch):
#         results.append(result["__data"][0])

#     assert len(results) == 2
    
#     for result in results:
#         assert "multimodal_data" in result
#         assert "messages" in result


@pytest.mark.asyncio
async def test_prepare_multimodal_udf_no_multimodal_content():
    udf = PrepareMultimodalUDF(
        data_column="__data", 
        expected_input_keys=["messages"],
        model="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    batch = {
        "__data": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            }
        ]
    }

    results = []
    async for result in udf(batch):
        results.append(result["__data"][0])

    assert len(results) == 1
    assert "multimodal_data" in results[0]
    assert results[0]["multimodal_data"] == {}
    assert "messages" in results[0]


def test_prepare_multimodal_udf_expected_keys():
    udf = PrepareMultimodalUDF(
        data_column="__data", 
        expected_input_keys=["messages"],
        model="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    assert udf.expected_input_keys == {"messages"}


# @pytest.mark.asyncio
# async def test_prepare_multimodal_udf_mixed_content_types():
#     udf = PrepareMultimodalUDF(
#         data_column="__data", 
#         expected_input_keys=["messages"],
#         model="Qwen/Qwen2.5-VL-3B-Instruct"
#     )

#     # Test batch with mixed content types
#     batch = {
#         "__data": [
#             {
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user", 
#                         "content": [
#                             {"type": "text", "text": "Analyze this content:"},
#                             {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}},
#                             {"type": "video_url", "video_url": {"url": "https://content.pexels.com/videos/free-videos.mp4"}}
#                         ]
#                     }
#                 ]
#             }
#         ]
#     }

#     results = []
#     async for result in udf(batch):
#         results.append(result["__data"][0])

#     assert len(results) == 1
#     assert "multimodal_data" in results[0]
#     assert "messages" in results[0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])