import json
import sys
import tempfile
import unittest
import pickle
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.multimodal.image_qa import mathvision_qwen3vl
from benchmark.multimodal.model_adapters import qwen3_vl
from sparsevllm.engine.sequence import Sequence
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.models.qwen3_vl import Qwen3VLModel


class Qwen3VLMathVisionDriverTest(unittest.TestCase):
    def test_default_system_prompt_matches_hf_reference(self):
        with patch.object(
            sys,
            "argv",
            [
                "mathvision_qwen3vl.py",
                "--model_path",
                "/tmp/model",
                "--dataset_path",
                "/tmp/data.json",
                "--save_path",
                "/tmp/out.jsonl",
            ],
        ):
            parsed = mathvision_qwen3vl.parse_args()

        self.assertEqual(
            parsed.system_prompt,
            "You are a helpful assistant suitable for solving math problems.",
        )
        self.assertEqual(parsed.submission_batch_size, 0)

    def test_select_tasks_respects_batch_resume_and_samples(self):
        dataset = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "results.jsonl"
            save_path.write_text(json.dumps({"sample_uid": "a__sample0"}) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                start=0,
                end=1,
                limit=-1,
                samples_per_item=2,
                resume=True,
                save_path=str(save_path),
                shard_rank=0,
                shard_world_size=1,
            )

            tasks = mathvision_qwen3vl.select_tasks(dataset, args)

        self.assertEqual(
            tasks,
            [
                (0, 1, "a__sample1"),
                (1, 0, "b__sample0"),
                (1, 1, "b__sample1"),
            ],
        )
        self.assertEqual(list(mathvision_qwen3vl.iter_batches(tasks, 2)), [tasks[:2], tasks[2:]])

    def test_select_tasks_supports_round_robin_shards(self):
        dataset = [{"id": str(idx)} for idx in range(10)]
        args = SimpleNamespace(
            start=0,
            end=9,
            limit=-1,
            samples_per_item=1,
            resume=False,
            save_path="",
            shard_rank=2,
            shard_world_size=4,
        )

        tasks = mathvision_qwen3vl.select_tasks(dataset, args)

        self.assertEqual(tasks, [(2, 0, "2"), (6, 0, "6")])

    def test_select_tasks_supports_deterministic_assignment(self):
        dataset = [{"id": str(idx)} for idx in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            assignment_path = Path(tmp) / "assignment.json"
            assignment_path.write_text(
                json.dumps({"world_size": 2, "assignments": [[7, 1, 4], [6, 0, 3, 2, 5]]}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                start=0,
                end=7,
                limit=-1,
                samples_per_item=1,
                resume=False,
                save_path="",
                shard_rank=0,
                shard_world_size=2,
                shard_assignment_path=str(assignment_path),
            )

            tasks = mathvision_qwen3vl.select_tasks(dataset, args)

        self.assertEqual(tasks, [(7, 0, "7"), (1, 0, "1"), (4, 0, "4")])

    def test_construct_messages_preserves_text_only_question(self):
        item = {
            "messages": [
                {"content": "Question: What is 1+1?"},
                {"content": "2"},
            ],
            "images": [],
        }

        messages = mathvision_qwen3vl.construct_messages(
            item,
            system_prompt="system",
            image_cache={},
            max_pixels=None,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        user_text = messages[1]["content"][0]["text"]
        self.assertIn("Question: What is 1+1?", user_text)
        self.assertIn("Put the final answer ONLY inside \\boxed{}.", user_text)

    def test_generate_batch_uses_single_batched_generate_call(self):
        class FakeProcessor:
            class Tokenizer:
                eos_token_id = 2
                pad_token_id = 0

            tokenizer = Tokenizer()

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.long),
                    "attention_mask": torch.ones((2, 3), dtype=torch.long),
                }

            def batch_decode(self, tokens, **kwargs):
                self.decoded_tokens = tokens
                return [",".join(str(token) for token in row) for row in tokens]

        class FakeModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[0, 11, 12, 101, 102], [21, 22, 23, 201, 202]], dtype=torch.long)

        processor = FakeProcessor()
        model = FakeModel()
        args = SimpleNamespace(
            max_new_tokens=2,
            temperature=0.0,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )

        responses = mathvision_qwen3vl.generate_batch(
            model,
            processor,
            messages=[[{"role": "user", "content": []}], [{"role": "user", "content": []}]],
            args=args,
            device=torch.device("cpu"),
        )

        self.assertEqual(responses, ["101,102", "201,202"])
        self.assertEqual(processor.kwargs["padding"], True)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["input_ids"].shape[0], 2)

    def test_sparsevllm_wrapper_splits_multimodal_inputs(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        class FakeEngine:
            def generate(self, prompts, sampling_params, use_tqdm=False):
                self.prompts = prompts
                self.sampling_params = sampling_params
                return [{"token_ids": [101]} for _ in prompts]

        engine = FakeEngine()
        wrapper = qwen3_vl.SparseQwen3VLGenerationWrapper(
            engine,
            FakeProcessor(),
            SimpleNamespace(image_token_id=99, vision_start_token_id=98),
        )
        output = wrapper.generate(
            input_ids=torch.tensor([[0, 98, 99, 99, 99, 99, 7]], dtype=torch.long),
            attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            pixel_values=torch.arange(4 * 6, dtype=torch.float32).view(4, 6),
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertEqual(engine.prompts[0]["prompt_token_ids"], [98, 99, 99, 99, 99, 7])
        mm_data = engine.prompts[0]["multi_modal_data"]
        self.assertEqual(tuple(mm_data["pixel_values"].shape), (4, 6))
        self.assertEqual(mm_data["image_grid_thw"].tolist(), [[1, 2, 2]])
        self.assertEqual(engine.sampling_params.max_tokens, 1)

    def test_sequence_ipc_preserves_multimodal_inputs(self):
        multi_modal_data = {
            "pixel_values": torch.arange(12).view(2, 6),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        sequence = Sequence(
            [1, 2, 3],
            SamplingParams(max_tokens=1),
            multi_modal_data=multi_modal_data,
        )

        restored = pickle.loads(pickle.dumps(sequence))

        self.assertTrue(
            torch.equal(
                restored.multi_modal_data["pixel_values"],
                multi_modal_data["pixel_values"],
            )
        )
        self.assertEqual(restored.multi_modal_data["image_grid_thw"].tolist(), [[1, 1, 2]])

    def test_qwen3vl_model_frees_request_visual_cache(self):
        model = object.__new__(Qwen3VLModel)
        model._seq_caches = {3: {"image_embeds": torch.ones(1)}, 5: {"image_embeds": torch.ones(1)}}

        Qwen3VLModel.free_seq(model, 3)

        self.assertNotIn(3, model._seq_caches)
        self.assertIn(5, model._seq_caches)

    def test_sparsevllm_futurekv_uses_native_engine_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge_path = Path(tmp) / "judge_model.pt"
            judge_path.touch()
            args = SimpleNamespace(
                method="futurekv",
                kv_budget=32,
                step_drop=16,
                window_size=64,
                divide_length=8,
                judge_state_path=str(judge_path),
                futurekv_num_full_layers=2,
                batch_size=3,
                vllm_max_num_seqs=4,
                vllm_gpu_memory_utilization=0.8,
                vllm_tensor_parallel_size=2,
                sparsevllm_chunk_prefill_size=128,
                vllm_max_model_len=4096,
                torch_dtype="bfloat16",
            )

            engine_kwargs = qwen3_vl.build_sparsevllm_engine_kwargs(args)

        self.assertEqual(engine_kwargs["sparse_method"], "futurekv")
        self.assertEqual(engine_kwargs["futurekv_window_size"], 64)
        self.assertEqual(engine_kwargs["tensor_parallel_size"], 2)
        self.assertEqual(engine_kwargs["engine_prefill_chunk_size"], 128)
        self.assertEqual(engine_kwargs["futurekv_judge_path"], str(judge_path))

if __name__ == "__main__":
    unittest.main()
