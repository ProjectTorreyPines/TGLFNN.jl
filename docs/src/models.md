# Trained models

````@example models
models = String[] #hide
for file in readdir(joinpath(dirname(dirname(@__DIR__)), "models")) #hide
    if !endswith(file, ".bson") #hide
        continue #hide
    end #hide
    push!(models, replace(file, ".bson"=>"")) #hide
end #hide
println(join(models,"\n")) #hide
````