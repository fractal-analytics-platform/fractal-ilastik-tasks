VERSION="0.1.1"
COMMMAND="mamba"
# If ENVPREFIX is not NULL, the environment will be created with the prefix $ENVPREFIX/$ENVNAME 
# If ENVPREFIX is NULL, the environment will be created in the default location
ENVPREFIX="NULL" 

# Do NOT change the following lines
ENVNAME=ilastik-tasks-$VERSION
PYTHON="python=3.9"

if [ "$ENVPREFIX" == "NULL" ]; then
    LOCATION="--name $ENVNAME"
else
    LOCATION="--prefix $ENVPREFIX/$ENVNAME"
fi

echo "Creating environment $ENVNAME with $PYTORCH_PACKAGE"
$COMMMAND run $COMMMAND create $LOCATION \
                               --override-channels \
                               -c pytorch \
                               -c ilastik-forge \
                               -c conda-forge $PYTHON ilastik \
                               --no-channel-priority --yes

echo "Installing plantseg-tasks version $VERSION"
$COMMMAND run --name $ENVNAME pip install git+https://github.com/fractal-analytics-platform/fractal-ilastik-tasks@$VERSION

echo "Downloading the __FRACTAL_MANIFEST__.json file file"
curl -O https://raw.githubusercontent.com/fractal-analytics-platform/fractal-ilastik-tasks/$VERSION/__FRACTAL_MANIFEST__.json
