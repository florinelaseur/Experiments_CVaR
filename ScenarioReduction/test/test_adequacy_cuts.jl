using TestItems: @testitem

@testitem "cut coefficients and rhs (MW / availability units)" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    df = DataFrame(;
        scenario=[1],
        timestep=[5],
        solar=[0.2],
        wind_onshore=[0.3],
        wind_offshore=[0.4],
        demand=[1000.0],
    )
    params = AdequacyParams(; peak_demand=1.5, hydro_cap=0.1, ens_cap=2.0)
    cuts = build_adequacy_cuts(df, 1, params)

    # One hour → one (non-dominated) cut.
    @test size(cuts.A) == (1, length(INVESTABLE_ASSETS))
    # Order: [ccgt, ocgt, solar, wind, wind_offshore, electrolizer, battery].
    @test cuts.A[1, :] == [1.0, 1.0, 0.2, 0.3, 0.4, 0.0, 0.05]
    # rhs = peak_demand·demand − hydro_cap − ens_cap.
    @test cuts.b[1] ≈ 1.5 * 1000.0 - 0.1 - 2.0
    @test cuts.timesteps == [5]
    @test cuts.scenario == 1
end

@testitem "non-dominated reduction drops dominated hours" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    #   h1: hard  (demand 100, low availability everywhere)
    #   h2: easy  (demand  80, high availability)          -> dominated by h1
    #   h3: hard  (demand 120 but also high availability)  -> not comparable to h1
    demand  = [100.0, 80.0, 120.0]
    a_solar = [0.1, 0.5, 0.6]
    a_won   = [0.1, 0.5, 0.6]
    a_woff  = [0.1, 0.5, 0.6]
    keep = _nondominated_indices(demand, a_solar, a_won, a_woff)
    @test sort(keep) == [1, 3]
end

@testitem "non-dominated reduction keeps one of exact duplicates" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    demand  = [100.0, 100.0]
    a_solar = [0.2, 0.2]
    a_won   = [0.2, 0.2]
    a_woff  = [0.2, 0.2]
    keep = _nondominated_indices(demand, a_solar, a_won, a_woff)
    @test keep == [1]                      # lowest index survives
end

@testitem "passes_adequacy is a sound reject filter" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    df = DataFrame(;
        scenario=[1], timestep=[1],
        solar=[0.2], wind_onshore=[0.3], wind_offshore=[0.4], demand=[1000.0],
    )
    params = AdequacyParams(; peak_demand=1.5, hydro_cap=0.1, ens_cap=2.0)
    cuts = build_adequacy_cuts(df, 1, params)
    rhs = cuts.b[1]                                  # = 1497.9

    # ccgt only, below the bar → provably infeasible.
    @test passes_adequacy([1000.0, 0, 0, 0, 0, 0, 0], cuts) == false
    # ccgt + battery clear the bar → not rejected.
    @test passes_adequacy([1500.0, 0, 0, 0, 0, 0, 500.0], cuts) == true
    # electrolyzer has zero supply coefficient: it must not help.
    @test passes_adequacy([0, 0, 0, 0, 0, 1e9, 0], cuts) == false
    # exactly meeting rhs with ccgt passes (within tolerance).
    @test passes_adequacy([rhs, 0, 0, 0, 0, 0, 0], cuts) == true
end

@testitem "adequacy_verdict reports the first binding scenario" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    df = DataFrame(;
        scenario=[1, 2], timestep=[1, 1],
        solar=[0.0, 0.0], wind_onshore=[0.0, 0.0], wind_offshore=[0.0, 0.0],
        demand=[668.0, 1000.0],
    )
    params = AdequacyParams(; peak_demand=1.5, hydro_cap=0.1, ens_cap=2.0)
    cuts1 = build_adequacy_cuts(df, 1, params)       # rhs ≈ 999.9
    cuts2 = build_adequacy_cuts(df, 2, params)       # rhs ≈ 1497.9
    cuts_list = [cuts1, cuts2]

    # ccgt = 1000 clears scenario 1 but not scenario 2.
    v = adequacy_verdict([1000.0, 0, 0, 0, 0, 0, 0], cuts_list)
    @test v.passed == false
    @test v.binding_scenario == 2

    v2 = adequacy_verdict([1600.0, 0, 0, 0, 0, 0, 0], cuts_list)
    @test v2.passed == true
    @test v2.binding_scenario == 0
end

@testitem "max-of-optima centre satisfies every scenario's cuts" setup = [AdequacyCutsSetup] tags = [:adequacy, :unit] begin
    df = DataFrame(;
        scenario=[1, 2], timestep=[1, 1],
        solar=[0.0, 0.0], wind_onshore=[0.0, 0.0], wind_offshore=[0.0, 0.0],
        demand=[668.0, 1000.0],
    )
    params = AdequacyParams(; peak_demand=1.5, hydro_cap=0.1, ens_cap=2.0)
    cuts_list = [build_adequacy_cuts(df, 1, params), build_adequacy_cuts(df, 2, params)]

    # Each scenario's "optimum" satisfies its own cut (ccgt ≥ rhs of that scenario).
    per_scenario = DataFrame(;
        scenario=[1, 2],
        ccgt=[1000.0, 1500.0],
        ocgt=[0.0, 0.0],
        solar=[0.0, 0.0],
        wind=[0.0, 0.0],
        wind_offshore=[0.0, 0.0],
        electrolizer=[0.1, 0.1],
        battery=[0.0, 0.0],
    )
    center = feasibility_center_max_optima(per_scenario, [1, 2])
    @test center == [1500.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]   # element-wise max
    @test adequacy_verdict(center, cuts_list).passed == true # monotone ⇒ passes all
end
