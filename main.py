import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import time
import csv
import glob
from run_SSAN import run_SSAN
from run_edgar import get_filings
from run_html_cleaner import run_html_cleaner
from preprocess_spacy import get_all_json

parser = argparse.ArgumentParser("Document-Level Relation Extraction from 10K-report by using SSAN")

""" Basic """
parser.add_argument("--name_companies", nargs="+")
parser.add_argument("--start_date", nargs="+")
parser.add_argument("--end_date", nargs="+")

# parser.add_argument("--save_raw_html", action="store_true")
# parser.add_argument("--save_processed_txt", action="store_true")
# parser.add_argument("--save_json", action="store_true")

parser.add_argument("--base_dirname", type=str, default="./output/")

""" Anything if you want """

""" SSAN """
parser.add_argument("--SSAN_model", type=str, default="roberta")
parser.add_argument("--SSAN_type_model", type=str, default="base", help="base / large")
parser.add_argument("--SSAN_entity_structure", type=str, default="biaffine", help="none / decomp / biaffine")
args = parser.parse_args()

def main(args):
    start = time.time()

    """ Basic Setting """
    print ("=" * 100)
    for arg in vars(args):
        print("\t {} : {}".format(arg, getattr(args, arg)))
    print ("=" * 100)

    if args.name_companies == None:
        name_companies = []
        with open('nasdaq.csv') as csvfile:
            reader = csv.reader(csvfile)
            for idx, line in enumerate(reader):
                if idx > 1:
                    name_companies.append(line[0])
    else:
        name_companies = args.name_companies

    # ---------------------------------
    if not os.path.isdir(args.base_dirname):
        os.mkdir(args.base_dirname)

    get_filings(name_companies=name_companies, start_date=args.start_date, end_date=args.end_date, download_dir=args.base_dirname)

    for idx, name_company in enumerate(name_companies):
        print (f"{idx:5}/{len(name_companies)} | {name_company}")
        print ("-")

        targetPattern = rf"{args.base_dirname}/{name_company}/10-K/*.txt"
        lst_paths = glob.glob(targetPattern)

        for path in lst_paths:
            print('start html cleaner!')
            _, file_name = run_html_cleaner(path=path, download=True)

            print('start json converter!')
            print('-'*10,file_name)
            inputs_json, json_name = get_all_json(file_name=file_name, download=True)
            print(f'Inputs json: {inputs_json}')
            print(f'Json name: {json_name}')

            print('start document RE using SSAN!')
            run_SSAN(parser=parser, input_json=inputs_json, SSAN_model=args.SSAN_model, SSAN_type_model=args.SSAN_type_model, SSAN_entity_structure=args.SSAN_entity_structure, save_dirname=f"{args.base_dirname}/{name_company}/")

    print (f"Elapsed Time : {time.time() - start}")

if __name__ == "__main__":
    main(args)