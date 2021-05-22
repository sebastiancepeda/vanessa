export PYTHONPATH=$(pwd) && \
echo PYTHONPATH: $PYTHONPATH && \
source venv/bin/activate && \
pytest && \
echo "End!"

