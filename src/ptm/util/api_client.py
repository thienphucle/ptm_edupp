import json
import os
import time
from typing import Optional, Union, Type, TypeVar

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel

from .logging_context import get_current_logger

# --- Constants ---
MODEL_LITE = "gpt-4o-mini"  
MODEL_PRO = "gpt-4o"      

_openai_client: Optional[OpenAI] = None

T = TypeVar("T", bound=BaseModel)


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Khởi tạo (lazy) OpenAI client.
    SDK tự động retry các lỗi 429, 5xx và timeout với exponential backoff.
    """
    global _openai_client
    if _openai_client is not None and api_key is None:
        return _openai_client

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        for d in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
            env_path = os.path.join(d, ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "OPENAI_API_KEY=" in line:
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            if key:
                break

    if not key:
        raise ValueError(
            "Do not find OPENAI_API_KEY. "
            "Please set it in environment variables or in a .env file."
        )

    _openai_client = OpenAI(
        api_key=key,
        max_retries=3, 
        timeout=60.0,
    )
    return _openai_client


def query_api_with_retries(
    prompt: str,
    model: str = MODEL_LITE,
    response_model: Optional[Type[T]] = None,
    outjson: bool = True,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Optional[Union[dict, list, str, BaseModel]]:
    """
    Gọi OpenAI Chat Completions API.

    Args:
        prompt: Nội dung prompt.
        model: Tên model (mặc định gpt-4o-mini).
        response_model: (Tùy chọn) Pydantic class. Nếu truyền vào, SDK sẽ
                        tự động dùng Structured Outputs và trả về instance
                        đã được parse. Đây là cách KHUYẾN NGHỊ.
        outjson: Nếu True và không có response_model, dùng JSON mode
                 (response_format={"type": "json_object"}).
        temperature: 0.0 để output ổn định, phù hợp cho extraction.
        max_tokens: Giới hạn output.
        api_key: Override API key nếu cần.

    Returns:
        - Nếu response_model được truyền: instance của Pydantic model đó.
        - Nếu outjson=True: dict hoặc list đã parse.
        - Nếu outjson=False: string thô.
        - None nếu thất bại sau tất cả các lần retry.
    """
    logger = get_current_logger()

    try:
        client = get_openai_client(api_key)
    except ValueError as e:
        logger.error(str(e))
        return None

    messages = [{"role": "user", "content": prompt}]

    try:

        if response_model is not None:
            logger.info(
                f"[OpenAI] Structured Output | model={model} | schema={response_model.__name__}"
            )
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                refusal = completion.choices[0].message.refusal
                logger.warning(f"[OpenAI] Model refused: {refusal}")
                return None
            logger.info("[OpenAI] Structured output parsed successfully.")
            return parsed

        if outjson:
            logger.info(f"[OpenAI] JSON mode | model={model}")
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            try:
                parsed = json.loads(content)
                logger.info("[OpenAI] JSON parsed successfully.")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"[OpenAI] JSON decode failed: {e}. Raw: {content[:200]}")
                return None

        logger.info(f"[OpenAI] Text mode | model={model}")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        logger.info("[OpenAI] Text response received.")
        return content

    except RateLimitError as e:
        logger.error(f"[OpenAI] Rate limit exceeded: {e}")
        return None
    except APITimeoutError as e:
        logger.error(f"[OpenAI] Request timed out: {e}")
        return None
    except APIConnectionError as e:
        logger.error(f"[OpenAI] Connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"[OpenAI] Unexpected error: {e}", exc_info=True)
        return None