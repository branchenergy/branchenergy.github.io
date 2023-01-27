---
title: Testing pydantic with Hypothesis
autor: Matt Badger
date: 2023-01-27
categories: python, testing
---

This post covers a few important points for generating valid and complete
[pydantic](https://docs.pydantic.dev/) model instances using
[Hypothesis](https://hypothesis.readthedocs.io/en/latest/). I'll assume you're familiar
with at least pydantic and won't go into detail on Hypothesis either: in short, it
enables _property-based_ testing, in which you specify the _type_ of something (a string,
an integer, or a pydantic model), and it generates valid data for that model. If this is
all new to you, [Rodrigo Girão Serrão wrote a really good article on this only last week](https://semaphoreci.com/blog/property-based-testing-python-hypothesis-pytest).

## Hypothesis + Pydantic

Hypothesis can be used to generate Pydantic models, however it is pretty fickle in places and it’s important to understand its limitations; in particular, making sure that the generated models cover the range of possible valid models.

We will use the following example model:

```python
class Thing(BaseModel):
    maybe_string: Optional[str] = Field(alias="maybeString")
    string_or_float: Union[str, float] = Field(alias="stringOrFloat")
    float_or_string: Union[float, str] = Field(alias="floatOrString")
    non_nan_float: confloat(allow_inf_nan=False) = Field(alias="nonNanFloat")
```

We can then create instances of this model for testing using Hypothesis, using the strategy builds, i.e. with `st.builds(Thing)`. builds takes keyword arguments matching field names whose values are separate strategies for generating valid data for their field.


## Use Field Aliases in st.builds

In the model above, `st.builds(Thing, maybe_string=st.text())` will not generate values for `maybe_string`! You must passs the alias instead, i.e. `st.builds(Thing, maybeString=st.text())`


## Optional Variables and Type Casting

If a variable is optional, it will always generate `None`, and if all fields are optional then Hypothesis will only generate one example instance whose values are all `None`. To get solve for this, you can use the `one_of` strategy, as follows:

```python
@given(
    st.builds(Thing, maybeString=st.one_of(st.none(), st.text()))
)
```

In addition to this, in the case of `Union`, Hypothesis will only generate instance of the first type for any types that can be cast as that type. So in the example above, `string_or_float` will always have type str because a float can be cast as a string; conversely `float_or_string` will generate both types. Solving this uses the same `one_of` strategy as above.


## Invalid Data

In the case of Pydantic constrained types, Hypothesis will generate data which are not valid and your test will raise an error immediately, so `non_nan_float` may be `nan` or `inf`. In this case, you must also pass a strategy which does generate valid data, using for example `st.floats(allow_nan=False)`.


## Addendum: Using Real-World Data

In addition to the above gotchas, it's sometimes useful to use real-world data in
Hypothesis. We have a number of microservices whose data models include addresses, and
these addresses have been validated as existing ahead of time, so we don't want
Hypothesis to go wild with `st.text`, but at the same time pass address components
in different parts of a pydantic model. In these cases, we have the following
composite strategy for generating valid models from a list of known addresses.

```python
@st.composite
def address_strategy(draw: Callable) -> RequestWithAddress:
    base_instance = draw(st.builds(RequestWithAddress))
    address_line_1, city, state, zipcode = draw(st.sampled_from(real_addresses))
    base_instance.address_line_1 = address_line_1
    base_instance.city = city
    base_instance.state = state
    base_instance.zipcode = zipcode

    return base_instance
```

Here, `real_addresses` is a `List[Tuple[str, str, str, str]]` of real addresses, which
we draw from using `sampled_from` and use to overwrite the data that Hypothesis
generates.
