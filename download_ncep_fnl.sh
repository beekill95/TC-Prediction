#! /bin/bash

# **NOTE**: requires bash version >= 4.2
#
# bash script to download selected files from rda.ucar.edu using Wget
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
# Command line arguments:
# ./download_ncep_fnl.sh "<output_directory>" "<begin_date>" ["<end_date>"]
# where <begin_date> and <end_date> follow the format: "yyyymmdd hh"
#
# <end_date> is optional, if <end_date> is not specified, it is assumed
# to get till the current date.
# Experienced Wget Users: add additional command-line flags here
#   Use the -r (--recursive) option with care
#   Do NOT use the -b (--background) option - simultaneous file downloads
#       can cause your data access to be blocked
opts="-N"

#
# Check input date range
#
DATE_FORMAT="%Y%m%d %H"
START_DATE=$(date --date="$2" +"$DATE_FORMAT" -u)
[ -z "$3" ] && END_DATE=$(date +"$DATE_FORMAT" -u) || END_DATE=$(date --date="$3" +"$DATE_FORMAT" -u)
printf "Download data from %s to %s\n" "$START_DATE" "$END_DATE"

#
# For the purpose of this script,
# we will only download observation data from May to November each year.
MONTHS_TO_KEEP=("05" "11")

#
# Get username and password
#
unset username
unset passwd

echo -n "Username: "
read username
echo -n "Password: "
read -s passwd
echo ""

num_chars=$(echo "$passwd" | awk '{print length($0)}')
if [[ $num_chars == 0 ]]; then
    echo "You need to set your password before you can continue"
    echo "  see the documentation in the script"
    exit
fi
newpass=""
while read -n1 c; do
  if [[ "$c" == "&" ]]; then
    c="%26"
  elif [[ "$c" == "?" ]]; then
    c="%3F"
  elif [[ "$c" == "=" ]]; then
    c="%3D"
  fi
  newpass="$newpass$c"
done <<< "$passwd"
passwd="$newpass"

#
cert_opt=""
# If you get a certificate verification error (version 1.10 or higher),
# uncomment the following line:
#set cert_opt = "--no-check-certificate"
#

# Base root of the server
BASE_URL="https://rda.ucar.edu"

#
# authenticate - NOTE: You should only execute this command ONE TIME.
# Executing this command for every data file you download may cause
# your download privileges to be suspended.
AUTH_FILE="auth_status.rda.ucar.edu"
COOKIES="auth.rda.ucar.edu.cookies"
wget $cert_opt -O $AUTH_FILE --save-cookies $COOKIES --post-data="email=$username&passwd=$passwd&action=login" "$BASE_URL/cgi-bin/login" -nv

#
# download the file(s)
# NOTE:  if you get 403 Forbidden errors when downloading the data files, check
#        the contents of the file 'auth_status.rda.ucar.edu'

# Download observation every 6h hour increment.
HOUR_INCREMENT=6

# Download observation data. 
OBSERVATION_DATE="$START_DATE"
CURRENT_YEAR=""
DATA_ROOT="$BASE_URL/data/OS/ds083.2/grib2"
while [[ "$OBSERVATION_DATE" < "$END_DATE" || "$OBSERVATION_DATE" == "$END_DATE"  ]]; do
    year=$(date --date="$OBSERVATION_DATE" +"%Y" -u)
    month=$(date --date="$OBSERVATION_DATE" +"%m" -u)

    # Create a separate directory for each year.
    if [[ "$CURRENT_YEAR" != "$year" ]]; then
        CURRENT_YEAR="$year"
        YEAR_OUTPUT_DIR="$1/$CURRENT_YEAR"
        [ -d "$YEAR_OUTPUT_DIR" ] || mkdir -p "$YEAR_OUTPUT_DIR"
    fi

    # Only download observations within the months we want.
    if [[ ("$month" > "${MONTHS_TO_KEEP[0]}" || "$month" == "${MONTHS_TO_KEEP[0]}")
        && ("$month" < "${MONTHS_TO_KEEP[1]}" || "$month" == "${MONTHS_TO_KEEP[1]}") ]]; then
        url="$DATA_ROOT/$year/$year.$month/fnl_$(date --date="$OBSERVATION_DATE" +"%Y%m%d_%H_00" -u).grib2"
        echo "Downloading observation $OBSERVATION_DATE from $url"
        wget $cert_opt $opts --load-cookies "$COOKIES" -P "$YEAR_OUTPUT_DIR" --progress=bar:force "$url" 2>&1 | tail -f -n +8
    else
        echo "SKIP downloading observation in $OBSERVATION_DATE"
    fi

    # Increment to the next observation.
    echo $OBSERVATION_DATE
    OBSERVATION_DATE=$(date --date="$OBSERVATION_DATE +$HOUR_INCREMENT hour" +"$DATE_FORMAT" -u)
done

#
# clean up
rm "$AUTH_FILE" "$COOKIES"

