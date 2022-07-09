import textutils as TU

INCL_LABELS =  ['Drug', 'Frequency','Reason','ADE']
HISTORY_SECTION = "History of Present Illness"
COMPLAINT_SECTION = "Chief Complaint"
DIAGNOSIS_SECTION = "Discharge Diagnosis"
def getConstants():
  return {
    "HISTORY_SECTION":HISTORY_SECTION,
    "COMPLAINT_SECTION":COMPLAINT_SECTION,
    "DIAGNOSIS_SECTION":DIAGNOSIS_SECTION,
    "INCL_LABELS":INCL_LABELS
  }
def parseRAnnotation(text):
  tmps = text.split('\t')
  if len(tmps)<2:
    return None
  rid = tmps[0]
  tmps[1] = tmps[1].replace("\n","")
  if len(tmps[1].split(" "))<3:
    return None
  label, arg1, arg2 = tmps[1].split(" ")
  arg1 = arg1.replace("Arg1:","")
  arg2 = arg1.replace("Arg2:","")
  return [rid, label,arg1, arg2]

def parseTAnnotation(text):
    words = text.split("\t")
    if len(words)!=3:
        return None
    ann_type, label_idx, text = words
    label, start, end = ("","","")
    # cases of line splits indicated by semicolons - just take whole span
    if ';' in label_idx:
        label_idx_filtered = []
        for el in label_idx.split():
            if ';' in el:
                continue
            label_idx_filtered.append(el)
        label, st, end = label_idx_filtered
    else:
        label, st, end = label_idx.split()
    if label in INCL_LABELS:
        return [ann_type,st, end, label,text.replace("\n","")]
    else:
        return None

# functions to filter the required sections from the script
def inRequiredSection(line, required_sections =None):
  if required_sections is None:
    required_sections=[HISTORY_SECTION,DIAGNOSIS_SECTION,COMPLAINT_SECTION]
  for r in required_sections:
    if r.lower() in line.lower():
      return True
  return False

def inNeighbourSection(line):
  neighbour_sections = ["Past Medical History:","Discharge Condition:","Major Surgical or Invasive Procedure:"]
  for r in neighbour_sections:
    if r.lower() in line.lower():
      return True
  return False

def getRequiredSections(text, requiredSections =None):

  textRows = text.split("\n")
  accept = False
  tmp = []
  for line in textRows:
    if inNeighbourSection(line):
      accept = False
      # tmp.append("\n")
    elif inRequiredSection(line, requiredSections):
      accept = True
    if accept==True:
      tmp.append(line)
  noteTxt = (' ').join(tmp)
  cleaned = TU.cleanText(noteTxt)
  return cleaned
