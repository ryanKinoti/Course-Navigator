from django.views.generic import TemplateView, ListView, DetailView
from .models import Course


class HomeView(TemplateView):
    template_name = 'home.html'


class CourseListView(ListView):
    model = Course
    template_name = 'core/course_list.html'
    context_object_name = 'courses'


class CourseDetailView(DetailView):
    model = Course
    template_name = 'core/course_details.html'
    context_object_name = 'course'
