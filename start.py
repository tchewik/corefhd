import argparse
import importlib

from isanlp.nlp_service_server import NlpServiceServer

parser = argparse.ArgumentParser(description='NLP service.')
parser.add_argument('-p', type = int, default = 3333, help = 'Port to listen.')
parser.add_argument('-t', type = int, default = 1, help = 'Number of workers.')
parser.add_argument('-m', type = str, default = 'isanlp.pipeline_default', help = 'Python module.')
parser.add_argument('-a', type = str, default= 'create_pipeline', help = 'Function for pipeline creation.')
parser.add_argument('--no_multiprocessing', type=bool, default=True,
                    help='Disable usage of multiprocessing.Pool, use direct call instead.')
args = parser.parse_args()

module_name = args.m
creator_fn_name = args.a
port = args.p
nthreads = args.t
no_multiprocessing = args.no_multiprocessing

creator_fn = getattr(importlib.import_module(module_name), creator_fn_name)
ppl = creator_fn(delay_init = True)
ppls = {ppl._name : ppl}

NlpServiceServer(ppls=ppls, port=port, max_workers=nthreads,
                 no_multiprocessing=no_multiprocessing).serve()