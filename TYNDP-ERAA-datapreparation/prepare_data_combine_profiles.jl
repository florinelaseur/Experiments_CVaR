using DataFrames
using CSV

cd(@__DIR__)


function combine_profiles()

    input_data_folder = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
    )

    demand_file = joinpath(
        input_data_folder,
        "profiles-demand.csv",
    )

    availability_file = joinpath(
        input_data_folder,
        "profiles-availability.csv",
    )

    inflow_file = joinpath(
        input_data_folder,
        "profiles-inflow.csv",
    )


    # ============================================================
    # CHECK INPUT FILES
    # ============================================================

    for file in [
        demand_file,
        availability_file,
        inflow_file,
    ]
        if !isfile(file)
            error("Required profile file does not exist: $file")
        end
    end


    # ============================================================
    # READ
    # ============================================================

    demand = CSV.read(
        demand_file,
        DataFrame,
    )

    availability = CSV.read(
        availability_file,
        DataFrame,
    )

    inflow = CSV.read(
        inflow_file,
        DataFrame,
    )


    # ============================================================
    # COMBINE
    # ============================================================

    all_profiles_df = vcat(
        demand,
        availability,
        inflow,
    )

    sort!(
        all_profiles_df,
        [
            :milestone_year,
            :scenario,
            :profile_name,
            :timestep,
        ],
    )


    # ============================================================
    # CHECK FOR DUPLICATE PROFILE ENTRIES
    # ============================================================

    duplicate_check = combine(
        groupby(
            all_profiles_df,
            [
                :milestone_year,
                :scenario,
                :profile_name,
                :timestep,
            ],
        ),
        nrow => :count,
    )

    duplicates = filter(
        :count => >(1),
        duplicate_check,
    )

    if !isempty(duplicates)
        error(
            """
            Duplicate profile entries found.

            Each combination of
            milestone_year / scenario / profile_name / timestep
            must occur exactly once.

            First duplicates:
            $(first(duplicates, min(10, nrow(duplicates))))
            """
        )
    end


    # ============================================================
    # WRITE LONG FORMAT
    # ============================================================

    profiles_file = joinpath(
        input_data_folder,
        "profiles.csv",
    )

    CSV.write(
        profiles_file,
        all_profiles_df;
        writeheader=true,
    )


    # ============================================================
    # CREATE WIDE FORMAT
    # ============================================================

    df_wide = unstack(
        all_profiles_df,
        [
            :milestone_year,
            :timestep,
            :scenario,
        ],
        :profile_name,
        :value,
    )

    profiles_wide_file = joinpath(
        input_data_folder,
        "profiles-wide-all-scenarios.csv",
    )

    CSV.write(
        profiles_wide_file,
        df_wide;
        writeheader=true,
    )


    # ============================================================
    # INFO
    # ============================================================

    println()
    println("Combined profiles written to:")
    println(profiles_file)

    println()
    println("Wide profiles written to:")
    println(profiles_wide_file)

    println()
    println(
        "Total number of profiles: ",
        length(unique(all_profiles_df.profile_name)),
    )
end


combine_profiles()