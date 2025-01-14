# custom_filters.py
from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    return dictionary.get(str(key))

@register.filter
def split(value, delimiter):
    """
       Split the string by the given delimiter.

       :param value: The string to split.
       :param delimiter: The delimiter to split by.
       :return: A list of substrings.
       """
    return value.split(delimiter)
