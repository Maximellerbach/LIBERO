"""Paired VLA-JEPA evaluation; change only MuJoCo's import path between runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/libero4390-pr-config")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model
from safetensors import safe_open

import libero.libero as libero
libero._assets_path_cache = "/home/maxime/github/libero-init-investigation/assets"
import mujoco
from libero.libero.envs.predicates.base_predicates import On
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from lerobot.sim.integrations.libero import LiberoEnv, _get_suite, canonical_observation

ROOT = Path(__file__).resolve().parent


def placement(env):
    sim = env._env.sim
    model, data = sim.model._model, sim.data._data
    bowl = model.body("akita_black_bowl_1_main").id
    ramekin = model.body("glazed_rim_porcelain_ramekin_1_main").id
    delta = data.xpos[bowl] - data.xpos[ramekin]
    contacts = [float(c.dist) for c in data.contact
                if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == {bowl, ramekin}]
    states = env._env.env.object_states_dict
    return {"on_ramekin": bool(On()(states["akita_black_bowl_1"], states["glazed_rim_porcelain_ramekin_1"])),
            "xy_cm": float(np.linalg.norm(delta[:2]) * 100),
            "dz_cm": float(delta[2] * 100), "contacts": len(contacts),
            "bowl_pos": data.xpos[bowl].tolist(), "ramekin_pos": data.xpos[ramekin].tolist()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    args = parser.parse_args()
    assert args.episodes % args.batch_size == 0
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "episodes.jsonl").exists(), "Use a fresh output directory"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    models = json.loads((ROOT / "models.json").read_text())
    model_path = Path(models["policy"]["path"])
    cfg = VLAJEPAConfig.from_pretrained(model_path)
    cfg.device = "cuda"
    cfg.qwen_model_name = models["qwen"]["path"]
    # These modules compute a training-only auxiliary loss; predict_action never uses them.
    # Keep all prompt dimensions and action/inference weights unchanged.
    cfg.enable_world_model = False
    print(f"Loading policy for MuJoCo {mujoco.__version__}", flush=True)
    policy = VLAJEPAPolicy(cfg)
    missing, unexpected = load_model(policy, str(model_path / "model.safetensors"), strict=False, device="cpu")
    assert not missing, missing
    alias = "model.qwen.model.model.language_model.embed_tokens.weight"
    verified_alias = False
    if alias in unexpected:
        # safetensors reports a redundant key for Qwen's tied input/output embeddings.
        qwen = policy.model.qwen.model
        assert qwen.get_input_embeddings().weight is qwen.get_output_embeddings().weight
        with safe_open(model_path / "model.safetensors", framework="pt") as file:
            assert torch.equal(file.get_tensor(alias), file.get_tensor("model.qwen.model.lm_head.weight"))
            assert torch.equal(qwen.get_input_embeddings().weight.cpu(), file.get_tensor(alias))
        unexpected.remove(alias)
        verified_alias = True
    assert all(k.startswith(("model.video_encoder.", "model.video_predictor.")) for k in unexpected), [k for k in unexpected if not k.startswith(("model.video_encoder.", "model.video_predictor."))]
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=model_path,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    print(f"Loaded all inference weights; skipped {len(unexpected)} auxiliary world-model tensors", flush=True)

    suite = _get_suite("libero_spatial")
    envs = [LiberoEnv(task_suite=suite, task_id=5, task_suite_name="libero_spatial",
                     obs_type="pixels_agent_pos", observation_width=256, observation_height=256,
                     episode_index=i, n_envs=args.batch_size, hard_reset=True)
            for i in range(args.batch_size)]
    state_path = Path("/home/maxime/github/hf-libero-task5-init/libero/libero/init_files/libero_spatial") / (envs[0].task + ".pruned_init")
    assert sha(state_path) == "2d2c50b7dcbf861fffa94d576f217ed34153e57487010d7a109a220d4d97189b"
    assert len(envs[0]._init_states) == 50
    manifest = {"models": models, "mujoco": mujoco.__version__, "mujoco_module": mujoco.__file__,
                "lerobot_commit": "b31254e4f624502b5c12163650d0c8d1dbe6eaa9", "hf_libero_base": "6422ca924565cb5f16a7f4c850bf3aaac2a3954e",
                "task_id": 5, "suite": "libero_spatial", "task": envs[0].task,
                "instruction": envs[0].task_description, "initial_state_file": str(state_path),
                "initial_state_sha256": sha(state_path), "script_sha256": sha(__file__),
                "policy_weights_sha256": sha(model_path / "model.safetensors"),
                "batch_size": args.batch_size, "episodes": args.episodes, "max_steps": args.max_steps,
                "resolution": [256, 256], "settle_steps": 10, "control_hz": 20,
                "action_chunk": cfg.chunk_size, "action_steps": cfg.n_action_steps,
                "noise_seed_formula": "1000000 + initial_state_id * 1000 + chunk_index; torch CPU generator",
                "auxiliary_world_model_loaded": False, "missing_inference_weights": list(missing),
                "skipped_auxiliary_weight_count": len(unexpected),
                "verified_redundant_embedding_alias": verified_alias,
                "packages": {k: importlib.metadata.version(k) for k in
                             ["torch", "torchvision", "transformers", "diffusers", "numpy", "robosuite", "safetensors"]},
                "gpu": torch.cuda.get_device_name()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    results = []
    try:
        for offset in range(0, args.episodes, args.batch_size):
            ids = list(range(offset, offset + args.batch_size))
            observations, starts, state_hashes = [], [], []
            for env, state_id in zip(envs, ids, strict=True):
                assert env.init_state_id == state_id
                state_hashes.append(hashlib.sha256(np.asarray(env._init_states[state_id]).tobytes()).hexdigest())
                obs, _ = env.reset(seed=state_id)
                observations.append(obs)
                starts.append(placement(env))
                Image.fromarray(canonical_observation(obs)["observation.images.image"]).save(args.out / f"state-{state_id:02d}-start.png")
            print(f"RESET {ids}: on_ramekin={[s['on_ramekin'] for s in starts]}", flush=True)
            policy.reset()
            pre.reset()
            post.reset()
            finished = [False] * args.batch_size
            successes = [False] * args.batch_size
            lengths = [0] * args.batch_size
            rewards = [0.] * args.batch_size
            action_histories = [[] for _ in ids]
            frame_histories = [[] for _ in ids]
            for step in range(args.max_steps):
                if step % cfg.n_action_steps == 0:
                    canon = [canonical_observation(obs) for obs in observations]
                    batch = prepare_canonical_observation({k: np.stack([c[k] for c in canon]) for k in canon[0]})
                    batch["task"] = [envs[0].task_description] * args.batch_size
                    batch = pre(batch)
                    chunk_index = step // cfg.n_action_steps
                    noise = torch.stack([torch.randn((cfg.chunk_size, cfg.action_dim),
                              generator=torch.Generator().manual_seed(1000000 + state_id * 1000 + chunk_index))
                              for state_id in ids]).to("cuda")
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(batch, noise=noise)
                    assert chunk.shape == (args.batch_size, cfg.chunk_size, 7), chunk.shape
                    assert torch.isfinite(chunk).all()
                actions = post(chunk[:, step % cfg.n_action_steps]).detach().cpu().numpy()
                assert np.isfinite(actions).all()
                for i, env in enumerate(envs):
                    if finished[i]:
                        continue
                    if ids[i] in (0, 1, 2) and step % 2 == 0:
                        frame_histories[i].append(canonical_observation(observations[i])["observation.images.image"])
                    observations[i], reward, terminated, truncated, info = env.step(actions[i])
                    action_histories[i].append(actions[i].copy())
                    rewards[i] += float(reward)
                    lengths[i] += 1
                    successes[i] = successes[i] or bool(info["is_success"])
                    finished[i] = bool(terminated or truncated or step + 1 == args.max_steps)
                    if finished[i]:
                        print(f"EPISODE state={ids[i]} success={successes[i]} steps={lengths[i]}", flush=True)
                if all(finished):
                    break
                if (step + 1) % 70 == 0:
                    print(f"PROGRESS states={ids} step={step+1} elapsed={time.monotonic()-started:.1f}s", flush=True)
            for i, state_id in enumerate(ids):
                end = placement(envs[i])
                row = {"initial_state_id": state_id, "initial_state_array_sha256": state_hashes[i],
                       "success": successes[i], "steps": lengths[i], "return": rewards[i],
                       "timeout": not successes[i] and lengths[i] == args.max_steps,
                       "start": starts[i], "end": end}
                results.append(row)
                with (args.out / "episodes.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                np.save(args.out / f"state-{state_id:02d}-actions.npy", np.asarray(action_histories[i]), allow_pickle=False)
                Image.fromarray(canonical_observation(observations[i])["observation.images.image"]).save(args.out / f"state-{state_id:02d}-end.png")
                if frame_histories[i]:
                    np.savez_compressed(args.out / f"state-{state_id:02d}-frames.npz", frames=np.stack(frame_histories[i]))
            print(f"BATCH_DONE {len(results)}/{args.episodes} successes={sum(r['success'] for r in results)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    finally:
        for env in envs:
            env.close()
    summary = {"mujoco": mujoco.__version__, "episodes": len(results), "successes": sum(r["success"] for r in results),
               "success_rate": float(np.mean([r["success"] for r in results])),
               "start_on_ramekin": sum(r["start"]["on_ramekin"] for r in results),
               "elapsed_seconds": time.monotonic()-started}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
