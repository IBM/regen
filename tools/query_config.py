""" Query variable value from a configargparse .cfg file
"""
import configargparse
import opts


def main(args):

    d = vars(args)
    # print(d)
    if args.var in d:
        print(d[args.var])
    else:
        raise ValueError(f"'{args.var}' not in cfg {args.config}")


if __name__ == "__main__":

    # Parsing arguments
    p = configargparse.ArgParser(description='config file arguments')

    # require config file
    p.add("-c", "--config", required=True, is_config_file=True, help="config file path")
    p.add("-v", "--var", required=True, help="variable we want .cfg value from")
    p = opts.add_arguments(p)

    args = p.parse_args()

    main(args)
