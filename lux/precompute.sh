#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=QUEUE
#SBATCH --account=QUEUE
#SBATCH --job-name=pre_stageSTAGE_noisy_zspec_runit_noiip
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/pre_stageSTAGE_noisy_zspec_runit_noiip.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stacks/
python precompute.py STAGE --noisy --zspec --runit --noiip'

if [[ $1 != [0-5] ]]; then
  echo "The first command line argument must be an int representing the stage."
  return 1
fi

STAGE=$1
shift

NOISY=false
ZSPEC=false
RUNIT=false
NOIIP=false
QUEUE=cpuq

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
    ?*)
      echo "Unknown option $1"
      return 1
      ;;
    *)
      break
  esac
  shift
done

echo "Submitting scripts..."
echo "stage: $STAGE"
echo "noisy: $NOISY"
echo "zspec: $ZSPEC"
echo "runit: $RUNIT"
echo "noiip: $NOIIP"
echo "queue: $QUEUE"

echo "Proceed? (yes/no)"

read PROCEED

if [ "$PROCEED" == 'yes' ]; then

  SCRIPT="${TEMPLATE//STAGE/$STAGE}"
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

  FILE=pre_stage${STAGE}.sh

  echo "$SCRIPT" > $FILE
  sbatch $FILE
  rm $FILE

fi
