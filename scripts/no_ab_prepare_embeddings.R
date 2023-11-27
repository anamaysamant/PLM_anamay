library(Platypus)
library(yaml)

print("Extracting sequence from VGM without AB")
args = commandArgs(trailingOnly=TRUE)


con <- file("config.yml", "r")
settings = read_yaml(con)

load(settings[["data_path"]])

if("vgm_colname" %in% names(settings)){
    col = settings[["vgm_colname"]]
} else{
    message("Please provide the column with the sequences for the PLM")
}

sequences = vgm$VDJ[col]
names(sequences) = "sequence"
write.csv(sequences, paste("outfiles/",args[1],"/clones.csv",sep=""))