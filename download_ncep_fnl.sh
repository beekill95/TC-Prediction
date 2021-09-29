#! /bin/bash
#
# bash script to download selected files from rda.ucar.edu using Wget
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
# Experienced Wget Users: add additional command-line flags here
#   Use the -r (--recursive) option with care
#   Do NOT use the -b (--background) option - simultaneous file downloads
#       can cause your data access to be blocked
opts="-N"
#
# Replace xxxxxx with your rda.ucar.edu password on the next uncommented line
# IMPORTANT NOTE:  If your password uses a special character that has special
#                  meaning to csh, you should escape it with a backslash
#                  Example:  set passwd = "my\!password"

unset username
unset passwd

echo -n "Username: "
read username
echo -n "Password: "
read -s passwd
echo ""

num_chars=$(echo "$passwd" | awk '{print length($0)}')
echo $num_chars
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

#
# authenticate - NOTE: You should only execute this command ONE TIME.
# Executing this command for every data file you download may cause
# your download privileges to be suspended.
AUTH_FILE="auth_status.rda.ucar.edu"
COOKIES="auth.rda.ucar.edu.cookies"
wget $cert_opt -O $AUTH_FILE --save-cookies $COOKIES --post-data="email=$username&passwd=$passwd&action=login" https://rda.ucar.edu/cgi-bin/login

#
# download the file(s)
# NOTE:  if you get 403 Forbidden errors when downloading the data files, check
#        the contents of the file 'auth_status.rda.ucar.edu'
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.cookies https://rda.ucar.edu/data/OS/ds083.2/grib2/2007/2007.12/fnl_20071206_12_00.grib2
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.cookies https://rda.ucar.edu/data/OS/ds083.2/grib2/2007/2007.12/fnl_20071206_18_00.grib2
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.cookies https://rda.ucar.edu/data/OS/ds083.2/grib2/2007/2007.12/fnl_20071207_00_00.grib2
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.cookies https://rda.ucar.edu/data/OS/ds083.2/grib2/2007/2007.12/fnl_20071207_06_00.grib2

#
# clean up
rm "$AUTH_FILE" "$COOKIES"

