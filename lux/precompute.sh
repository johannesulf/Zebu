#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --job-name=precompute_lLENS_BIN_sSOURCE_BIN_stageSTAGE_SURVEY
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/precompute_lLENS_BIN_sSOURCE_BIN_stageSTAGE_SURVEY.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stage_STAGE/
python precompute.py --lens_bin=LENS_BIN --source_bin=SOURCE_BIN --survey=SURVEY'

if [[ $1 != [0-2] ]]; then
  echo "The first command line argument must be an int representing the stage."
  return 1
fi

if [ "$2" != "" ] && [ "$2" != "des" ] && [ "$2" != "hsc" ] && [ "$2" != "kids" ]; then
  echo "The second command line argument, if given, must be des, hsc or kids."
  return 1
fi

if [ "$2" != "" ] && [ "$1" == "0" ]; then
  echo "You cannot specify a survey for stage 0."
  return 1
fi

NL=4
NS=4

if [ "$2" == "kids" ]; then
  NS=5
fi

for (( i=0; i<$NL; i++ )); do
  for (( k=0; k<$NS; k++ )); do

    SCRIPT="${TEMPLATE//LENS_BIN/$i}"
    SCRIPT="${SCRIPT//SOURCE_BIN/$k}"
    SCRIPT="${SCRIPT//STAGE/${1}}"
    if [ "$2" == "" ]; then
      SCRIPT="${SCRIPT//_SURVEY/}"
      SCRIPT="${SCRIPT// --survey=SURVEY/}"
    else
      SCRIPT="${SCRIPT//SURVEY/${2}}"
    fi
    FILE=precompute_${i}_${k}_stage${1}_${2}.sh
    
    echo "$SCRIPT" > $FILE
    sbatch $FILE
    rm $FILE
  done
done
