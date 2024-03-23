# Generate the arenas
python -m run --generate --node-space 10 --probability-space 0.01 0.1

# Evaluate the algorithms on all the generated arenas
for file in arenas/*.pkl; do
    echo "Evaluating $file"
    python -m run --solve --arena="$file" --save-results
done
