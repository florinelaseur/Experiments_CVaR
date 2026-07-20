using CSV
using DataFrames

script_dir = @__DIR__
output_path = joinpath(script_dir, "..", "outputs")

vec = [1, 2]
output_file = joinpath(output_path, "results.csv")

function assert_unique_base_names!(df_current, df_new, file_path)
    common = intersect(df_current.base_name, df_new.base_name)

    # allow base_name == "0_HourlyBenchmark"
    allowed = Set(["0_HourlyBenchmark"])
    problematic = setdiff(common, allowed)

    if !isempty(problematic)
        error("ERROR: Found duplicate base_name(s) $(collect(problematic)) in file $file_path")
    end
end

merged = DataFrame()

for i in vec
    file_path = joinpath(output_path, "results$(i).csv")

    @info "Reading $file_path"
    df_i = CSV.read(file_path, DataFrame)

    # enforce no duplicates except the benchmark row
    if nrow(merged) > 0
        assert_unique_base_names!(merged, df_i, file_path)
    end

    # keep only one benchmark row 
    if nrow(merged) == 0
        # first file: keep all its rows
        append!(merged, df_i)
    else
        # remove benchmark row from this df_i
        df_clean = filter(row -> row.base_name != "0_HourlyBenchmark", df_i)
        append!(merged, df_clean)
    end
end

CSV.write(output_file, merged)
@info "Saved final merged results to $output_file"