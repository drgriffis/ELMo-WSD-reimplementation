'''
Prints a single setting from a .config file to stdout
'''

import sys
import configparser

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog CONFIG SECTION SETTING')
        (options, args) = parser.parse_args()
        if len(args) != 3:
            parser.print_help()
            exit()
        return args
    configf, section, key = _cli()

    config = configparser.ConfigParser()
    config.read(configf)
    sys.stdout.write('%s\n' % config[section][key])
