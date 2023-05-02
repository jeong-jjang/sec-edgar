import argparse
import json
import en_core_web_sm
from pprint import pprint
import os


def match_label(label):
    if label in ["PERSON"]:
        output = "PER"
    elif label in ["ORG"]:
        output = "ORG"
    elif label in ["NORP", "FAC", "GPE", "LOC"]:
        output = "LOC"
    elif label in ["DATE", "TIME"]:
        output = "TIME"
    elif label in ["PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
        output = "NUM"
    elif label in ["PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
        output = "MISC"
    else:
        raise NotImplementedError(f'{label}')
    return output

def check_overlap(vertexSet, text, target_dic):
    # vertexSet: list(list(dic))
    # text: str
    if len(vertexSet) != 0:
        for lst in vertexSet:
            for dic in lst:
                if dic["name"] == text:
                    lst.append(target_dic)
                    return vertexSet, True
    return vertexSet, False

def get_json_files(nlp, document=None, title="None"):

    if document is None:
        # document = "Kungliga Hovkapellet (The Royal Court Orchestra) is a Swedish orchestra, originally part of the Royal Court in Sweden's capital Stockholm. The orchestra originally consisted of both musicians and singers. It had only male members until 1727, when Sophia Schröder and Judith Fischer were employed as vocalists; in the 1850s, the harpist Marie Pauline Åhman became the first female instrumentalist. From 1731, public concerts were performed at Riddarhuset in Stockholm. Since 1773, when the Royal Swedish Opera was founded by Gustav III of Sweden, the Kungliga Hovkapellet has been part of the opera's company."
        document = "Our plan to grow the volume and profitability of our vehicles and energy storage products depends on significant lithium-ion battery cell production by our partner Panasonic at Gigafactory Nevada. Although Panasonic has a long track record of producing high-quality cells at significant volume at its factories in Japan, it 	has relatively limited experience with cell production at Gigafactory Nevada	, which began in 2017. Moreover, although Panasonic is co-located with us at Gigafactory Nevada, it is free to make its own operational decisions, such as its determination to temporarily suspend its manufacturing there in response to the COVID-19 pandemic. In addition, we produce several vehicle components, such as battery modules and packs incorporating the cells produced by Panasonic for Model 3 and Model Y and drive units (including to support Gigafactory Shanghai production), at Gigafactory Nevada, and we also manufacture energy storage products there. In the past, some of the manufacturing lines for certain product components took longer than anticipated to ramp to their full capacity, and additional bottlenecks may arise in the future as we continue to increase the production rate and introduce new lines. If we or Panasonic are unable to or otherwise do not maintain and grow our respective operations at Gigafactory Nevada production, or if we are unable to do so cost-effectively or hire and retain highly-skilled personnel there, our ability to manufacture our products profitably would be limited, which may harm our business and operating results."

    output = {"vertexSet": [],
              "title": title,
              "sents": []}

    sequences = nlp(document)
    for sent_id, sequence in enumerate(sequences.sents):
        # tokenize and propagate model
        doc = nlp(sequence.text)

        print([(X.text, X.label_) for X in doc.ents])
        print([(X, X.ent_iob_, X.ent_type_) for X in doc])

        for X in doc.ents:
            label = match_label(X.label_)
            temp_vertexSet = {"pos": [X.start, X.end], "type": label, "sent_id": int(sent_id), "name": X.text}
            output["vertexSet"], check = check_overlap(output["vertexSet"], X.text, temp_vertexSet)
            if not check:
                output["vertexSet"].append([temp_vertexSet])
        for XXX in doc:
            if XXX.text == 'we' or XXX.text == 'We':
                label = "ORG"
                temp_vertexSet = {"pos": [XXX.i, XXX.i + 1], "type": label, "sent_id": int(sent_id), "name": XXX.text}
                output["vertexSet"], check = check_overlap(output["vertexSet"], XXX.text, temp_vertexSet)
                if not check:
                    output["vertexSet"].append([temp_vertexSet])

        temp_sent = [XX.text for XX in doc]
        output["sents"].append(temp_sent)

    pprint(output)
    # with open("docRE_sample_tesla_10K.json", "w") as json_file:
    #     json.dump(output, json_file)
    return output

def get_all_json(file_name, file=None, download=True):
    json_name = "{}.json".format(file_name[:-4])
    if os.path.isfile(json_name):
        with open(json_name) as json_file:
            json_output = json.load(json_file)
            return json_output, json_name

    if file is None:
        input_text = open(file_name, 'r', encoding='utf-8')
    else:
        input_text = file

    json_output = []
    nlp = en_core_web_sm.load()
    for num_para, line in enumerate(input_text.readlines()):
        json_output.append(get_json_files(nlp=nlp, document=line, title=str(num_para)))

    if download:
        with open(json_name, "w") as json_file:
            json.dump(json_output, json_file)

    return json_output, json_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocumentRE with spacy')
    parser.add_argument('--file', default='example', type=str, help='text file name to inference')

    args = parser.parse_args()
    get_all_json(args.file)
