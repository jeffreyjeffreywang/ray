from pydantic import BaseModel

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig, SGLangEngineProcessorConfig

class AnswerWithExplain(BaseModel):
    problem: str
    answer: int
    explain: str

json_schema = AnswerWithExplain.model_json_schema()

chat_template = """
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
    """

processor_config = SGLangEngineProcessorConfig(
    model_source="unsloth/Llama-3.1-8B-Instruct",
    engine_kwargs=dict(
        context_length=2048,
        disable_cuda_graph=True,
        dtype="half",
    ),
    batch_size=16,
    concurrency=1,
    apply_chat_template=True,
    chat_template=chat_template,
    tokenize=True,
    detokenize=True,
)

processor = build_llm_processor(
    processor_config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system", "content": "You are a calculator"},
            {"role": "user", "content": f"{row['id']} ** 3 = ?"},
        ],
        sampling_params=dict(
            temperature=0.3,
            max_new_tokens=50,  # SGLang uses max_new_tokens instead of max_tokens
        ),
    ),
    postprocess=lambda row: {
        "resp": row["generated_text"],
    },
)

ds = ray.data.range(60)
ds = ds.map(lambda x: {"id": x["id"], "val": x["id"] + 5})
ds = processor(ds)
ds = ds.materialize()
outs = ds.take_all()