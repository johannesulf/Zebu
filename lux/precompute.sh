#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=QUEUE
#SBATCH --account=QUEUE
#SBATCH --job-name=pre_stageSTAGE_sSOURCE_BIN_noisy_zspec_runit_noiip
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/pre_stageSTAGE_sSOURCE_BIN_noisy_zspec_runit_noiip.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stacks/
python precompute.py STAGE SOURCE_BIN --noisy --zspec --runit --noiip'

if [[ $1 != [0-4] ]]; then
  echo "The first command line argument must be an int representing the stage."
  return 1
fi

STAGE=$1
shift

NOISY=false
ZSPEC=false
RUNIT=false
NOIIP=false
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
    -r|--runit)
      RUNIT=true
      ;;
    -i|--noiip)
      NOIIP=true
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
echo "sources: $SOURCE_BIN_MIN - $SOURCE_BIN_MAX"
echo "noisy: $NOISY"
echo "zspec: $ZSPEC"
echo "runit: $RUNIT"
echo "noiip: $NOIIP"
echo "queue: $QUEUE"

finished () {

  PRE_FINISHED=false

  LOG=log/pre_stage${STAGE}_s${SOURCE_BIN}_noisy_zspec_runit.out

  if [ "$NOISY" != true ]; then
    LOG="${LOG//_noisy/}"
  fi

  if [ "$ZSPEC" != true ]; then
    LOG="${LOG//_zspec/}"
  fi

  if [ "$RUNIT" != true ]; then
    LOG="${LOG//_runit/}"
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


for (( SOURCE_BIN=$SOURCE_BIN_MIN; SOURCE_BIN<=$SOURCE_BIN_MAX; SOURCE_BIN++ )); do

  finished

  let N_TOT++

  if $PRE_FINISHED; then
    let N_SUC++
  fi

done

if $OVERWRITE; then
  echo "Number of jobs: $N_TOT"
else
  echo "Finished jobs: $N_SUC"
  echo "Remaining jobs: $(expr $N_TOT - $N_SUC)"
fi

echo "Proceed? (yes/no)"

read PROCEED

if [ "$PROCEED" == 'yes' ]; then
  for (( SOURCE_BIN=$SOURCE_BIN_MIN; SOURCE_BIN<=$SOURCE_BIN_MAX; SOURCE_BIN++ )); do

    SCRIPT="${TEMPLATE//STAGE/$STAGE}"
    SCRIPT="${SCRIPT//SOURCE_BIN/$SOURCE_BIN}"
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

    if [ "$RUNIT" != true ]; then
      SCRIPT="${SCRIPT//_runit/}"
      SCRIPT="${SCRIPT// --runit/}"
    fi

    if [ "$NOIIP" != true ]; then
      SCRIPT="${SCRIPT//_noiip/}"
      SCRIPT="${SCRIPT// --noiip/}"
    fi

    FILE=pre_stage${STAGE}_${LENS_BIN}_${SOURCE_BIN}.sh

    finished

    if ! $PRE_FINISHED || $OVERWRITE; then
      echo "$SCRIPT" > $FILE
      sbatch $FILE
      rm $FILE
    fi

  done
fi
