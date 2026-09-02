# Confirms the custom material factory is registered in the built OpenSees.exe.
model BasicBuilder -ndm 3 -ndf 3

nDMaterial TimberHoffman3D 1 \
    2050.8 172.1 172.1 \
    0.45 0.45 0.50 \
    145.2 145.2 68.0 \
    35.0 2.5 2.5 \
    20.0 0.7 0.7 \
    5.0 5.0 0.5 \
    1200.0 270.0 0.60 0.50 \
    60.0 0.5 0.5 \
    1.0e-4 5.0 1.0

puts "TimberHoffman3D material registration passed."
wipe
