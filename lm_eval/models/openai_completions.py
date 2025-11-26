import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="auto",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        # Auto-detect tokenizer backend
        if tokenizer_backend == "auto":
            if base_url:
                from lm_eval.utils import check_remote_tokenizer_support

                if check_remote_tokenizer_support(
                    base_url,
                    verify_certificate=verify_certificate,
                    ca_cert_path=ca_cert_path,
                    auth_token=auth_token,
                ):
                    eval_logger.info(
                        "Auto-detected remote tokenizer support. Using remote tokenizer backend."
                    )
                    tokenizer_backend = "remote"
                else:
                    eval_logger.info(
                        "Remote tokenizer not supported. Using huggingface tokenizer backend."
                    )
                    tokenizer_backend = "huggingface"
            else:
                eval_logger.warning(
                    "No base_url provided. Using huggingface tokenizer backend."
                )
                tokenizer_backend = "huggingface"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    """
    Minimal chat-completions wrapper.
    - Only accepts messages as list[dict].
    - No tokenization or template logic.
    - Use with --apply_chat_template or ensure upstream formats messages correctly.
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), (
            "LocalChatCompletion expects messages as list[dict]. "
            "If you see this error, ensure --apply_chat_template is set or upstream code formats messages correctly."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

        # For loglikelihood tasks, request logprobs
        if not generate:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        return payload

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    tmp[choices["index"]] = choices["message"]["content"]
            except Exception as e:
                # account for cases that generation is blocked by content filter,
                # which is common for Azure OpenAI Service,
                # not sure if need to account for multiple choices
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """
        Parse logprobs from chat completions API response.
        Chat completions return logprobs in choice.logprobs.content format.
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out, ctxlen in zip(outputs, ctxlens):
            for choice in sorted(out["choices"], key=itemgetter("index")):
                assert ctxlen > 0, "Context length must be greater than 0"

                # Chat completions return logprobs as choice.logprobs.content
                # which is a list of ChatCompletionTokenLogprob objects
                logprobs_data = choice.get("logprobs")
                if logprobs_data is None or logprobs_data.get("content") is None:
                    eval_logger.warning(
                        "No logprobs found in response. Ensure logprobs=True in request."
                    )
                    res.append((0.0, False))
                    continue

                content_logprobs = logprobs_data["content"]

                # Skip the context tokens (first ctxlen tokens) and sum continuation logprobs
                # Note: we don't skip the last token since chat completions only returns
                # logprobs for the generated tokens, not including a future prediction
                continuation_logprobs = content_logprobs[ctxlen:]

                if not continuation_logprobs:
                    eval_logger.warning(
                        f"No continuation tokens found (ctxlen={ctxlen}, total={len(content_logprobs)})"
                    )
                    res.append((0.0, False))
                    continue

                # Sum the logprobs
                total_logprob = sum(
                    token_data["logprob"] for token_data in continuation_logprobs
                )

                # Check if greedy: for each token, check if it has the highest logprob
                # among its top alternatives
                is_greedy = True
                for token_data in continuation_logprobs:
                    current_logprob = token_data["logprob"]
                    top_logprobs = token_data.get("top_logprobs", [])

                    if top_logprobs:
                        # Check if current token has the max logprob
                        max_logprob = max(alt["logprob"] for alt in top_logprobs)
                        # Use small epsilon for float comparison
                        if abs(current_logprob - max_logprob) > 1e-6:
                            is_greedy = False
                            break

                res.append((total_logprob, is_greedy))

        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        from lm_eval.models.api_models import JsonChatStr

        # Handle JsonChatStr objects - extract the prompt string
        if isinstance(string, JsonChatStr):
            string = string.prompt

        # If we have a tokenizer, use it (needed for loglikelihood context lengths)
        if self.tokenizer is not None:
            return super().tok_encode(
                string,
                left_truncate_len=left_truncate_len,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )
        # Otherwise, just return the string as-is
        return string

    def _encode_pair(
        self, context: Union[str, Any], continuation: Union[str, Any]
    ) -> tuple[list, list]:
        """
        Override to handle JsonChatStr objects from chat templates.

        For chat completions with tokenizer, we compute token lengths but keep
        the JsonChatStr object for the API call.
        """
        import json

        from lm_eval.models.api_models import JsonChatStr

        # For JsonChatStr, we need special handling
        if isinstance(context, JsonChatStr):
            # Parse the chat messages
            context_messages = json.loads(context.prompt)

            # The continuation is a plain string that should be the assistant's response
            if isinstance(continuation, str):
                # Create a chat with the assistant response for tokenization
                combined_messages = context_messages + [
                    {"role": "assistant", "content": continuation, "type": "text"}
                ]
            elif isinstance(continuation, JsonChatStr):
                combined_messages = context_messages + json.loads(continuation.prompt)
            else:
                combined_messages = context_messages

            # If we have a tokenizer, use it to compute token counts
            if self.tokenizer is not None:
                if self.tokenizer_backend == "huggingface":
                    # Render both context and full prompt using chat template
                    context_str = self.tokenizer.apply_chat_template(
                        context_messages, tokenize=False, add_generation_prompt=True
                    )
                    full_str = self.tokenizer.apply_chat_template(
                        combined_messages, tokenize=False, add_generation_prompt=False
                    )
                    # Tokenize to get counts
                    context_enc = self.tokenizer.encode(
                        context_str, add_special_tokens=False
                    )
                    full_enc = self.tokenizer.encode(full_str, add_special_tokens=False)
                elif self.tokenizer_backend == "remote":
                    # Remote tokenizer doesn't support apply_chat_template
                    # We need to render the template ourselves using a simple format
                    # This is a fallback - ideally use HuggingFace tokenizer with the actual model's template

                    # Simple chat template rendering (Gemma/Llama style)
                    def simple_chat_template(messages, add_generation_prompt=False):
                        """Simple fallback chat template for remote tokenizer"""
                        result = ""
                        for msg in messages:
                            role = msg["role"]
                            content = msg["content"]
                            if role == "system":
                                result += (
                                    f"<start_of_turn>system\n{content}<end_of_turn>\n"
                                )
                            elif role == "user":
                                result += (
                                    f"<start_of_turn>user\n{content}<end_of_turn>\n"
                                )
                            elif role == "assistant":
                                result += (
                                    f"<start_of_turn>model\n{content}<end_of_turn>\n"
                                )

                        if add_generation_prompt:
                            result += "<start_of_turn>model\n"

                        return result

                    context_str = simple_chat_template(
                        context_messages, add_generation_prompt=True
                    )
                    full_str = simple_chat_template(
                        combined_messages, add_generation_prompt=False
                    )

                    # Use remote tokenizer to encode
                    context_enc = self.tok_encode(context_str)
                    full_enc = self.tok_encode(full_str)
                else:
                    raise ValueError(
                        f"Unsupported tokenizer_backend '{self.tokenizer_backend}' for chat completions with loglikelihood. "
                        "Use 'huggingface' or 'remote'."
                    )

                # Important: Store the JsonChatStr in the context encoding and lengths in continuation
                # We'll use a special marker: wrap the combined JsonChatStr and store token counts
                combined_json = JsonChatStr(
                    json.dumps(combined_messages, ensure_ascii=False)
                )

                # Return: [combined_json, len(context_enc)], [len(continuation)]
                # The length will be used to compute ctxlen
                # We'll override batch_loglikelihood_requests to handle this properly
                return [combined_json, len(context_enc)], [
                    len(full_enc) - len(context_enc)
                ]
            else:
                raise ValueError(
                    "LocalChatCompletion with apply_chat_template requires a tokenizer for loglikelihood tasks. "
                    "Please specify tokenizer_backend='huggingface' or 'remote', and provide a tokenizer if needed."
                )

        # For regular strings (shouldn't happen with --apply_chat_template)
        if self.tokenizer is not None:
            # Handle rstrip logic manually to avoid calling it on non-strings
            n_spaces = (
                len(context) - len(context.rstrip()) if isinstance(context, str) else 0
            )
            if n_spaces > 0:
                continuation = context[-n_spaces:] + continuation
                context = context[:-n_spaces]

            context_enc = self.tok_encode(context)
            whole_enc = self.tok_encode(context + continuation)
            continuation_enc = whole_enc[len(context_enc) :]
            return context_enc, continuation_enc
        else:
            return [context], [continuation]

    def batch_loglikelihood_requests(self, chunks):
        """
        Override to handle JsonChatStr objects properly.

        When we have JsonChatStr with tokenization, _encode_pair returns:
        context_enc = [JsonChatStr(...), context_token_len]
        continuation_enc = [continuation_token_len]

        We need to extract the JsonChatStr for the API call and the lengths for ctxlen.
        """
        from lm_eval.models.api_models import JsonChatStr

        inputs = []
        ctxlens = []
        cache_keys = []

        for chunk in chunks:
            for cache_key, context_enc, continuation_enc in chunk:
                # Check if this is a JsonChatStr request (special format)
                if len(context_enc) == 2 and isinstance(context_enc[0], JsonChatStr):
                    # Extract: [JsonChatStr, ctx_len], [cont_len]
                    combined_json = context_enc[0]
                    ctx_len = context_enc[1]
                    continuation_enc[0] if continuation_enc else 0

                    # For JsonChatStr, we append it directly (not wrapped in a list)
                    # model_call expects List[JsonChatStr] for batched chat messages
                    inputs.append(combined_json)
                    ctxlens.append(ctx_len)
                    cache_keys.append(cache_key)
                else:
                    # Normal token-based request
                    inp = (context_enc + continuation_enc)[-self.max_length :]
                    if len(inp) < len(context_enc + continuation_enc):
                        eval_logger.warning(
                            f"Context length ({len(context_enc)}) + continuation length ({len(continuation_enc)}) > max_length ({self.max_length}). Left truncating context."
                        )
                    ctxlen = len(context_enc) - max(
                        0, len(context_enc) + len(continuation_enc) - self.max_length
                    )

                    inputs.append(inp)
                    ctxlens.append(ctxlen)
                    cache_keys.append(cache_key)

        return inputs, ctxlens, cache_keys


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if (
            "o1" in self.model
            or "5" in self.model
            or "o3" in self.model
            or "o4" in self.model
        ):
            output.pop("stop")
            output["temperature"] = 1

        # For loglikelihood tasks, request logprobs
        if not generate:
            output["logprobs"] = True
            output["top_logprobs"] = 1

        return output


@register_model("azure-openai-chat-completions")
class AzureOpenaiChatCompletionsLM(OpenAIChatCompletion):
    def __init__(
        self,
        model: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        base_url: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        truncate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.truncate = truncate
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=self.api_key
        )

    @cached_property
    def api_key(self):
        key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `AZURE_OPENAI_API_KEY` environment variable."
            )
        return key
