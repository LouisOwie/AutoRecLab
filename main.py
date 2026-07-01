import asyncio
import os
from argparse import ArgumentParser

from config import get_config
from treesearch.search import TreeSearch
from utils.checks import require_executable
from treesearch.utils.available_datasets import get_datasets_table
from treesearch.utils.costs_tracker import get_cost_tracker, get_model_table
from utils.log import _ROOT_LOGGER, attach_file_handler, set_log_level
from utils.path import mkdir
from utils.statistics_tracker import get_statistics_tracker

logger = _ROOT_LOGGER.getChild("main")
cost_tracker = get_cost_tracker()
statistics_tracker = get_statistics_tracker()


async def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = get_config()
    out_dir = mkdir(config.out_dir)
    args = get_args()

    if handle_management_command(args, out_dir):
        return

    if args.model is not None:
        config.agent.code = config.agent.code.model_copy(update={"model": args.model})

    user_request = resolve_user_request(args)
    if user_request is None or user_request.strip() == "":
        logger.error(
            "No request provided. Please provide a prompt using --prompt or --prompt-file, or type it manually."
        )
        return

    log_user_request(out_dir, user_request, args.prompt_no_log)
    await run_autoreclab(user_request, config, out_dir)


def handle_management_command(args, out_dir) -> bool:
    if args.init:
        mkdir(out_dir / "workspace")
        return True

    if args.list_datasets:
        datasets_table = get_datasets_table()
        print(datasets_table)
        return True

    if args.list_models:
        models_table = get_model_table()
        print(models_table)
        return True

    return False


def resolve_user_request(args) -> str | None:
    if args.prompt is not None:
        return args.prompt

    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    user_req_lines: list[str] = []
    print('Enter you request, write "!start" to start:')
    while True:
        line = input("> ")
        if line.lower().strip().startswith("!start"):
            break
        user_req_lines.append(line)

    return "\n".join(user_req_lines)


def log_user_request(out_dir, user_request: str, prompt_no_log: bool):
    if prompt_no_log:
        return

    prompt_file = out_dir / "entered_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(user_request)


async def run_autoreclab(user_request: str, config, out_dir):
    attach_file_handler(out_dir)
    cost_tracker.set_out_dir(out_dir)
    statistics_tracker.set_out_dir(out_dir)
    require_executable("dot")

    logger.info("Starting AutoRecLab...")
    logger.debug(f"User request:\n{user_request}")
    ts = TreeSearch(user_request, config=config)
    await ts._async_init()
    await ts.run()

    cost_tracker.saveSummarized()
    statistics_tracker.summarize_statistics()


def get_args():
    parser = ArgumentParser("AutoRecLab")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--prompt-no-log", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
