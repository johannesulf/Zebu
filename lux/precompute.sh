#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=QUEUE
#SBATCH --account=QUEUE
#SBATCH --job-name=pre_stageSTAGE_lLENS_BIN_sSOURCE_BIN_noisy_zspec
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/pre_stageSTAGE_lLENS_BIN_sSOURCE_BIN_noisy_zspec.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stacks/
python precompute.py STAGE LENS_BIN SOURCE_BIN --noisy --zspec'

if [[ $1 != [0-2] ]]; then
  echo "The first command line argument must be an int representing the stage."
  return 1
fi

STAGE=$1
shift

NOISY=false
ZSPEC=false
LENS_BIN_MIN=0
LENS_BIN_MAX=3
SOURCE_BIN_MIN=0
SOURCE_BIN_MAX=4
QUEUE=cpuq
OVERWRITE=false

while :; do
  case $1 in
    -n|--noisy)
      NOISY=true
      ;;
    -z|--zspec)
      ZSPEC=true
      ;;
    -o|--overwrite)
      OVERWRITE=true
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
    ?*)
      echo "Unknown option $1"
      return 1
      ;;
    *)
      break
  esac
  shift
done

if [ $STAGE == 0 ]; then
  SOURCE_BIN_MAX=$(( SOURCE_BIN_MAX > 3 ? 3 : SOURCE_BIN_MAX ))
fi

echo "Submitting scripts..."
echo "stage: $STAGE"
echo "lenses: $LENS_BIN_MIN - $LENS_BIN_MAX"
echo "sources: $SOURCE_BIN_MIN - $SOURCE_BIN_MAX"
echo "noisy: $NOISY"
echo "zspec: $ZSPEC"
echo "queue: $QUEUE"

finished () {

  PRE_FINISHED=false

  LOG=log/pre_stage${STAGE}_l${LENS_BIN}_s${SOURCE_BIN}_noisy_zspec.out

  if [ "$NOISY" != true ]; then
    LOG="${LOG//_noisy/}"
  fi

  if [ "$ZSPEC" != true ]; then
    LOG="${LOG//_zspec/}"
  fi

  if test -f "$LOG"; then
      LAST_LINE=$(tail -1 $LOG)
      if [ "$LAST_LINE" == 'Finished successfully!' ]; then
        PRE_FINISHED=true
      fi
  fi

}

N_TOT=0
N_SUC=0


for (( LENS_BIN=$LENS_BIN_MIN; LENS_BIN<=$LENS_BIN_MAX; LENS_BIN++ )); do
  for (( SOURCE_BIN=$SOURCE_BIN_MIN; SOURCE_BIN<=$SOURCE_BIN_MAX; SOURCE_BIN++ )); do

    finished

    let N_TOT++

    if $PRE_FINISHED; then
      let N_SUC++
    fi

  done
done

if $OVERWRITE; then
  echo "Number of jobs: $N_TOT"
else
  echo "Finished jobs: $N_SUC"
  echo "Remaining jobs: $(expr $N_TOT - $N_SUC)"
fi

echo "Proceed?"

read PROCEED

if [ "$PROCEED" == 'yes' ]; then
  for (( LENS_BIN=$LENS_BIN_MIN; LENS_BIN<=$LENS_BIN_MAX; LENS_BIN++ )); do
    for (( SOURCE_BIN=$SOURCE_BIN_MIN; SOURCE_BIN<=$SOURCE_BIN_MAX; SOURCE_BIN++ )); do

      SCRIPT="${TEMPLATE//LENS_BIN/$LENS_BIN}"
      SCRIPT="${SCRIPT//SOURCE_BIN/$SOURCE_BIN}"
      SCRIPT="${SCRIPT//STAGE/$STAGE}"
      SCRIPT="${SCRIPT//QUEUE/$QUEUE}"

      if [ "$QUEUE" == "leauthaud" ]; then
        SCRIPT="${SCRIPT//#SBATCH --time=1-0:00:00/#SBATCH --time=7-0:00:00}"
      fi

      if [ "$NOISY" != true ]; then
        SCRIPT="${SCRIPT//_noisy/}"
        SCRIPT="${SCRIPT// --noisy/}"
      fi

      if [ "$ZSPEC" != true ]; then
        SCRIPT="${SCRIPT//_zspec/}"
        SCRIPT="${SCRIPT// --zspec/}"
      fi

      FILE=pre_stage${STAGE}_${LENS_BIN}_${SOURCE_BIN}.sh

      finished

      if ! $PRE_FINISHED || $OVERWRITE; then
        echo "$SCRIPT" > $FILE
        sbatch $FILE
        rm $FILE
      fi

    done
  done
fi
