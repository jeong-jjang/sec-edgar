import argparse
import json
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from SSAN.dataset import docred_convert_examples_to_features as convert_examples_to_features
from SSAN.dataset import DocREDProcessor

from SSAN.model import (BertForDocRED, RobertaForDocRED)
import pickle

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def predict(args, model, tokenizer, processor, save_dirname, prefix=""):
    pred_examples = processor.get_test_examples()
    label_map = processor.get_label_map()
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    eval_dataset = load_and_cache_examples(args, tokenizer, processor)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = inputs["ent_mask"].detach().cpu().numpy()
            out_label_ids = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, inputs["ent_mask"].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    print("eval_loss: {}".format(eval_loss))
    output_preds = []
    for (i, (example, pred, ent_mask)) in enumerate(zip(pred_examples, preds, ent_masks)):
        for h in range(len(example.vertexSet)):
            for t in range(len(example.vertexSet)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit in enumerate(pred[h][t]):
                    if predicate_id == 0:
                        continue
                    else:
                        output_preds.append((logit, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[0], reverse=True)
    output_preds_thresh = []
    for i in range(len(output_preds)):
        if output_preds[i][0] < args.predict_thresh:
            break
        output_preds_thresh.append({"title": output_preds[i][1],
                                    "h_idx": output_preds[i][2],
                                    "t_idx": output_preds[i][3],
                                    "r": output_preds[i][4],
                                    "evidence": []
                                    })
    # write pred file
    # if not os.path.exists('./data/DocRED/') and args.local_rank in [-1, 0]:
    #     os.makedirs('./data/DocRED')
    with open(f"{save_dirname}/SSAN_result_raw.json", 'w') as f:
        json.dump(output_preds_thresh, f)

    return output_preds_thresh


def load_and_cache_examples(args, tokenizer, processor, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data
    label_map = processor.get_label_map()

    examples = processor.get_test_examples()


    features = convert_examples_to_features(
        examples,
        args.model_type,
        tokenizer,
        max_length=args.max_seq_length,
        max_ent_cnt=args.max_ent_cnt,
        label_map=label_map
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_ent_mask = torch.tensor([f.ent_mask for f in features], dtype=torch.float)
    all_ent_ner = torch.tensor([f.ent_ner for f in features], dtype=torch.long)
    all_ent_pos = torch.tensor([f.ent_pos for f in features], dtype=torch.long)
    all_ent_distance = torch.tensor([f.ent_distance for f in features], dtype=torch.long)
    all_structure_mask = torch.tensor([f.structure_mask for f in features], dtype=torch.bool)
    all_label = torch.tensor([f.label for f in features], dtype=torch.bool)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance,
                            all_structure_mask, all_label, all_label_mask)

    return dataset

def run_SSAN(parser, input_json, SSAN_model, SSAN_type_model, SSAN_entity_structure, save_dirname):
    # parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )
    parser.add_argument("--no_naive_feature", action="store_true",
                        help="do not exploit naive features for DocRED, include ner tag, entity id, and entity pair distance")
    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=False,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.5, type=float, help="pred thresh")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--lr_schedule", default='linear', type=str, choices=['linear', 'constant'],
                        help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    #########################################################################
    args.model_type = SSAN_model
    args.entity_structure = SSAN_entity_structure
    args.model_name_or_path = f"./SSAN/pretrained_lm/{SSAN_model}_{SSAN_type_model}/"
    args.do_predict = True
    args.predict_thresh = 0.46544307
    args.checkpoint_dir = f"./SSAN/pretrained_SSAN/{SSAN_model}_{SSAN_type_model}#{SSAN_entity_structure}"
    #-----------------------------------------------------------------------#
    f = open("./SSAN/label_map.json", 'r')
    label_map = json.load(f)
    f.close()

    #########################################################################

    ModelArch = None
    if args.model_type == 'roberta':
        ModelArch = RobertaForDocRED
    elif args.model_type == 'bert':
        ModelArch = BertForDocRED

    if args.no_naive_feature:
        with_naive_feature = False
    else:
        with_naive_feature = True

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    processor = DocREDProcessor(input_json=input_json, label_map=label_map)
    label_map = processor.get_label_map()
    num_labels = len(label_map.keys())

    # Load pretrained model and tokenizer

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # predict
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        output_preds_thresh = predict(args, model, tokenizer, processor, save_dirname=save_dirname)

    test_data = input_json
    predict_data = output_preds_thresh

    f = open("./SSAN/rel_info.json", "r")
    rel_info = json.load(f)
    f.close()

    title2idx = dict()
    processed_predict_data = []
    for idx, ele in enumerate(test_data):
        onedocument_info = dict()
        title2idx[ele["title"]] = idx

        onedocument_info["title"] = ele["title"]
        onedocument_info["sents"] = ele["sents"]
        onedocument_info["entities"] = []
        onedocument_info["relation"] = []

        vertexSet = ele["vertexSet"]
        for entity_idx, entity_info in enumerate(vertexSet):
            entity_names = [d["name"] for d in entity_info]
            onedocument_info["entities"].append(entity_names)
        processed_predict_data.append(onedocument_info)

    for ele in predict_data:
        title = ele["title"]
        idx = title2idx[title]

        head_entity = processed_predict_data[idx]["entities"][ele["h_idx"]][0]
        tail_entity = processed_predict_data[idx]["entities"][ele["t_idx"]][0]

        relation = rel_info[ele["r"]]
        processed_predict_data[idx]["relation"].append((head_entity, tail_entity, relation))

    exist_relation = []
    for ele in processed_predict_data:
        if len(ele["relation"]) != 0:
            exist_relation.append(ele)

    f = open(f"{save_dirname}/SSAN_result_all_relation.p", "wb")
    pickle.dump(processed_predict_data, f)
    f.close()

    f = open(f"{save_dirname}/SSAN_result_exist_relation.p", "wb")
    pickle.dump(exist_relation, f)
    f.close()