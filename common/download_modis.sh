#!/bin/bash
tile=$1
output_dir=$2
header="Authorization: Bearer $BEARER_STRING"

mkdir ${output_dir}
for product in {'MOD17A2H','MYD17A2H'}
do
    if [ "${product:0:3}" == "MOD" ]; then 
    startyear=2000
    startyearday=49
    else
    startyear=2002
    startyearday=185
    fi
    echo ${startyear}


# Use the accept-regex to filter all in 1 go
    for((year=${startyear};year<=2021;year++))
    do
        if [ ${year} == ${startyear} ]; then
            day0=${startyearday}
        else
            day0=1
        fi
        for day in $(seq -f "%03g" ${day0} 8 365)
        do
        https_proxy="$PROXY_URL" wget -e robots=off -m -np -c -R .html,.tmp -nH --cut-dirs=4 --accept-regex "${product}.A${year}${day}.${tile}.006*" "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/${product}/${year}/${day}/" --header "${header}" -P ${output_dir}
        done
    done
done