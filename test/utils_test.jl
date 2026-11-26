
@testset "cvmcriterion()" begin
    import GriddedBlockMaxima.cvmcriterion
    @test cvmcriterion(Uniform(), [0.75, 0.25]) ≈ 1 / 48
    @test cvmcriterion(Uniform(), [0.5, 0.5])   ≈ 1 / 12 # Presence of ties
end