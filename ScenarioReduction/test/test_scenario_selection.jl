using TestItems: @testitem

@testitem "format_selected_scenarios sorts and space-joins" setup = [
    ExportSelectionSetup,
] tags = [:selection, :unit] begin
    @test format_selected_scenarios([12, 3, 231, 21]) == "3 12 21 231"
end

@testitem "format_selected_scenarios single and already-sorted" setup = [
    ExportSelectionSetup,
] tags = [:selection, :unit] begin
    @test format_selected_scenarios([7]) == "7"
    @test format_selected_scenarios([1, 2, 3]) == "1 2 3"
end
