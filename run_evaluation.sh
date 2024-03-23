# Generate the arenas
python -m run --generate --node-space 10 50 100 200 500 1000 5000 --probability-space 0.01 0.05 0.1 0.2 0.5

# Evaluate the algorithms on all the generated arenas
for file in arenas/*.pkl; do
    echo "Evaluating $file"
    python -m run --solve --arena="$file" --save-results
    python -m run --solve --arena="$file" --save-results --optimize
done
