import argparse

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Run the AI agent with command-line arguments."
    )

    # Add arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        help="Specify the mode in which to run the AI agent."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, the script will print additional information."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run the AI loop."
    )

    # Parse the arguments from sys.argv
    args = parser.parse_args()

    # Use the parsed arguments
    if args.verbose:
        print("[INFO] Verbose mode is ON")

    print(f"Starting AI agent in `{args.mode}` mode...")
    print(f"Running for {args.iterations} iteration(s)...")

    # Your main logic would go here
    # e.g., initialize or load your flow controller, LLM, etc.
    # Possibly start a simple CLI loop or server.

if __name__ == "__main__":
    main()
