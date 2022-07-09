# get data and format it into a csv
# author: Sowmya.R
import glob
import pandas as pd

import textutils as TU
import clinicalDataUtils as CU

data_dir = 'training_20180910'  # Path to directory containing .ann and .txt files
output_dir = 'output'
dataFiles =  glob.glob(data_dir+'/*.txt')
len(dataFiles)

HISTORY_SECTION = "History of Present Illness"
COMPLAINT_SECTION = "Chief Complaint"
DIAGNOSIS_SECTION = "Discharge Diagnosis"

all_clinicalNotes = []
all_annotations = []
for file in dataFiles:
    file_ann = []
    ann_labels = []
    segment = ""
    sections ={
        "patient":"",
        "history":"",
        "complaint":"",
        "diagnosis":"",
        "annotations":""
    }
    
    with open(file,'r') as f:
        textFileData = f.readlines()
        sections["patient"] = file.split("\\")[1].replace(".txt","")
        for line in textFileData:
            if HISTORY_SECTION.lower() in line.lower(): 
                segment = "history"
                continue
            elif COMPLAINT_SECTION.lower() in line.lower():
                segment="complaint"
                continue
            elif DIAGNOSIS_SECTION.lower() in line.lower():
                segment="diagnosis"
                continue
            elif CU.inNeighbourSection(line):
                segment = ""
            if segment=="" or line==" ":
                continue
            sections[segment] = sections[segment]+line.lower()
    
    sections["history"] = TU.cleanText(sections["history"])
    sections["complaint"] = TU.cleanText(sections["complaint"])
    sections["diagnosis"] = TU.cleanText(sections["diagnosis"])

    with open(file.replace('.txt','.ann'),'r') as f:
        annFileData = f.readlines()
        print("reading file {}".format(file.replace(".txt",".ann")))
        for line in annFileData:
            if line[0].upper() =="T":
                annotations = CU.parseTAnnotation(line)
                if annotations is not None:
                    all_annotations.append(annotations)
                    file_ann.append({"label":annotations[-2],"pattern":annotations[-1]})
    sections["annotations"]= file_ann
    
    
    if sections["diagnosis"]=="": #skip the ones that doesnot have diagnosis
        continue
    all_clinicalNotes.append(sections)


all_clinicalNotes_df = pd.DataFrame(all_clinicalNotes)
all_clinicalNotes_df.to_csv("all_notes.csv", index=False)
print(all_clinicalNotes_df.shape)