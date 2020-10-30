#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=QUEUE
#SBATCH --account=QUEUE
#SBATCH --job-name=precompute_lLENS_BIN_sSOURCE_BIN_stageSTAGE_SURVEY_gamma_zspec
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/precompute_lLENS_BIN_sSOURCE_BIN_stageSTAGE_SURVEY_gamma_zspec.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stage_STAGE/
python precompute.py --lens_bin=LENS_BIN --source_bin=SOURCE_BIN --survey=SURVEY --gamma --zspec'

if [[ $1 != [0-2] ]]; then
  echo "The first command line argument must be an int representing the stage."
  return 1
fi

STAGE=$1
shift

SURVEY=
GAMMA=false
ZSPEC=false
LENS_BIN_MIN=0
LENS_BIN_MAX=3
SOURCE_BIN_MIN=0
SOURCE_BIN_MAX=4
QUEUE=cpuq

while :; do
  case $1 in
    -s|--survey)
      if [ "$2" ]; then
        if [ "$2" != "des" ] && [ "$2" != "hsc" ] && [ "$2" != "kids" ]; then
          echo "The survey must be des, hsc or kids."
          return 1
        fi
        if [ $STAGE == "0" ]; then
          echo "You cannot give a survey for stage 0."
          return 1
        fi
        SURVEY=$2
        shift
      else
        echo 'ERROR: "--survey" requires a non-empty option argument.'
        return 1
      fi
      ;;
    -g|--gamma)
      GAMMA=true
      ;;
    -z|--zspec)
      ZSPEC=true
      ;;
    -q|--queue)
      if [ "$2" ]; then
        if [ "$2" != "leauthaud" ] && [ "$2" != "cpuq" ] && [ "$2" != "gpuq" ]; then
          echo "The queue must be leauthaud, cpuq or gpuq."
          return 1
        fi
        QUEUE=$2
        shift
      else
        echo 'ERROR: "--queue" requires a non-empty option argument.'
        return 1
      fi
      ;;
    --lmin)
      if [[ $2 == [0-3] ]]; then
        LENS_BIN_MIN=$2
        shift
      else
        echo 'ERROR: "--lmin" requires an integer less than 4.'
        return 1
      fi
      ;;
    --lmax)
      if [[ $2 == [0-3] ]]; then
        LENS_BIN_MAX=$2
        shift
      else
        echo 'ERROR: "--lmax" requires an integer less than 4.'
        return 1
      fi
      ;;
    --smin)
      if [[ $2 == [0-4] ]]; then
        SOURCE_BIN_MIN=$2
        shift
      else
        echo 'ERROR: "--smin" requires an integer less than 5.'
        return 1
      fi
      ;;
    --smax)
      if [[ $2 == [0-4] ]]; then
        SOURCE_BIN_MAX=$2
        shift
      else
        echo 'ERROR: "--smax" requires an integer less than 5.'
        return 1
      fi
      ;;
    -?*)
       echo "Unknown option $1"
       return 1
       ;;
    *)
      break
  esac
  shift
done

if [ "$SURVEY" != "kids" ]; then
  SOURCE_BIN_MAX=$(( SOURCE_BIN_MAX > 3 ? 3 : SOURCE_BIN_MAX ))
fi

echo "Submitting scripts..."
echo "stage: $STAGE"
echo "survey: $SURVEY"
echo "lenses: $LENS_BIN_MIN - $LENS_BIN_MAX"
echo "sources: $SOURCE_BIN_MIN - $SOURCE_BIN_MAX"
echo "gamma: $GAMMA"
echo "zspec: $ZSPEC"
echo "queue: $QUEUE"

for (( lens=$LENS_BIN_MIN; lens<=$LENS_BIN_MAX; lens++ )); do
  for (( source=$SOURCE_BIN_MIN; source<=$SOURCE_BIN_MAX; source++ )); do

    SCRIPT="${TEMPLATE//LENS_BIN/$lens}"
    SCRIPT="${SCRIPT//SOURCE_BIN/$source}"
    SCRIPT="${SCRIPT//STAGE/${STAGE}}"
    if [ "$SURVEY" == "" ]; then
      SCRIPT="${SCRIPT//_SURVEY/}"
      SCRIPT="${SCRIPT// --survey=SURVEY/}"
    else
      SCRIPT="${SCRIPT//SURVEY/${SURVEY}}"
    fi
    SCRIPT="${SCRIPT//QUEUE/${QUEUE}}"
    if [ "$GAMMA" != true ] ; then
      SCRIPT="${SCRIPT//_gamma/}"
      SCRIPT="${SCRIPT// --gamma/}"
    fi
    if [ "$ZSPEC" != true ] ; then
      SCRIPT="${SCRIPT//_zspec/}"
      SCRIPT="${SCRIPT// --zspec/}"
    fi
    FILE=precompute_${lens}_${source}_stage${STAGE}_${SURVEY}.sh
    
    echo "$SCRIPT" > $FILE
    sbatch $FILE
    rm $FILE
  done
done
