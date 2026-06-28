from src.models import Body
from src.geological_models import (
    create_single_body_model,
    create_two_body_model,
    create_three_body_model,
)

body1 = Body(
    x_center=2500,
    z_center=700,
    width=800,
    height=500,
    density=450,
    name="Ore Body",
)

body2 = Body(
    x_center=5200,
    z_center=1200,
    width=900,
    height=700,
    density=-300,
    name="Salt Dome",
)

body3 = Body(
    x_center=7000,
    z_center=1600,
    width=600,
    height=500,
    density=250,
    name="Basement",
)

single = create_single_body_model(
    x_center=3000,
    z_center=900,
    width=900,
    height=600,
    density=400,
)

two = create_two_body_model(body1, body2)

three = create_three_body_model(
    body1,
    body2,
    body3,
)

print(single)
print(two)
print(three)