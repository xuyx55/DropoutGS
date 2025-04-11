base=$1
mask_path=$2
scene=$3

for scan_id in $scene
do  
    if [ -d $base/$scan_id ]; then
        # rm -r $base/$scan_id/mask
        mkdir $base/$scan_id/mask
        id=0
        if [ -d ${mask_path}/$scan_id/mask ]; then
            for file in ${mask_path}/scan8/*
            do  
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp ${file//scan8/$scan_id'/mask'} $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done

            else

            for file in ${mask_path}/$scan_id/*
            do
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp $file $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done
        fi
    fi
    
done