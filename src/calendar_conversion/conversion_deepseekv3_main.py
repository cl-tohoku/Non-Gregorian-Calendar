import argparse

import conversion_deepseekv3_en
import conversion_deepseekv3_ja


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True, help="DeepSeek API key for DeepSeek V3")
    args = parser.parse_args()

    conversion_deepseekv3_en.run(args)
    conversion_deepseekv3_ja.run(args)


if __name__ == "__main__":
    main()
