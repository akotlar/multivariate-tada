FROM ubuntu:latest

RUN apt-get graphviz
RUN pip install graphviz pymc3 tensorflow tensorflow_probability torch pyro 