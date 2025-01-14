from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import render, redirect
from django.utils.decorators import method_decorator
from rest_framework.views import APIView

from apps.core.models import Subject, Course  # Import Subject from core app
from .models import StudentSubject, Recommendation  # Import from local models
from .serializers import (
    GradeInputSerializer, RecommendationSerializer,
    SubjectSelectionSerializer
)
from .services.recommender import CourseRecommender


class SubjectSelectionView(APIView):
    @method_decorator(login_required)
    def get(self, request):
        subjects = Subject.objects.filter(is_active=True)
        return render(request, 'recommendations/subject_selection.html', {
            'available_subjects': subjects
        })

    @method_decorator(login_required)
    def post(self, request):
        subject_ids = request.POST.getlist('selected_subjects')
        data = {'subject_ids': subject_ids}

        serializer = SubjectSelectionSerializer(data=data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    profile = request.user.userprofile
                    # Update UserProfile
                    serializer.update(profile, serializer.validated_data)

                    # Clear existing entries (optional)
                    StudentSubject.objects.filter(user=request.user).delete()

                    # Create new StudentSubject entries
                    student_subjects = [
                        StudentSubject(
                            user=request.user,
                            subject_id=subject_id,
                        ) for subject_id in subject_ids
                    ]
                    StudentSubject.objects.bulk_create(student_subjects)

                    messages.success(request, "Subjects successfully selected!")
                    return redirect('dashboard')

            except Exception as e:
                messages.error(request, "An error occurred while saving your selections.")
                return render(request, 'recommendations/subject_selection.html', {
                    'errors': {'system': str(e)},
                    'available_subjects': Subject.objects.filter(is_active=True)
                })

        return render(request, 'recommendations/subject_selection.html', {
            'errors': serializer.errors,
            'available_subjects': Subject.objects.filter(is_active=True)
        })


class DashboardView(APIView):
    @method_decorator(login_required)
    def get(self, request):
        profile = request.user.userprofile
        grades = StudentSubject.objects.filter(user=request.user)

        # Get top 3 recent recommendations
        recent_recommendations = Recommendation.objects.filter(user=request.user).order_by('-created_at')[:6]

        grade_breakdown = {}
        for grade in grades:
            grade_breakdown[grade.subject.name] = grade.grade

        context = {
            'selected_subjects': profile.selected_subjects.all(),
            'existing_grades': {g.subject.id: g.grade for g in grades},
            'grade_breakdown': grade_breakdown,
            'average_grade': profile.calculate_mean_grade(),
            'recent_recommendations': RecommendationSerializer(recent_recommendations, many=True).data
        }

        return render(request, 'dashboard.html', context)

    @method_decorator(login_required)
    def post(self, request):
        grades_data = []
        for key, value in request.POST.items():
            if key.startswith("grade_"):  # Check for keys with 'grade_'
                subject_id = key.split("_")[1]  # Extract the subject ID (after 'grade_')
                grades_data.append({"subject_id": int(subject_id), "grade": value})

        serializer = GradeInputSerializer(data=grades_data, many=True)

        if serializer.is_valid():
            try:
                # Get the validated data
                grades_data = serializer.validated_data

                # Maintain separate dictionaries for IDs and names
                subject_grades_by_id = {}
                subject_grades_by_name = {}

                for grade_entry in grades_data:
                    subject = Subject.objects.get(id=grade_entry['subject_id'])
                    subject_grades_by_id[grade_entry['subject_id']] = grade_entry['grade']
                    subject_grades_by_name[subject.name] = grade_entry['grade']

                # Save grades to StudentSubject using IDs
                for subject_id, grade in subject_grades_by_id.items():
                    StudentSubject.objects.update_or_create(
                        user=request.user,
                        subject_id=subject_id,
                        defaults={'grade': grade}
                    )

                # Use names for recommender
                recommender = CourseRecommender()
                recommendations = recommender.predict_courses(subject_grades_by_name)

                # Delete existing recommendations for this user
                Recommendation.objects.filter(user=request.user).delete()

                # Save new recommendations
                created_recommendations = []
                for rec in recommendations:
                    try:
                        course = Course.objects.get(code=rec['course'])  # Assuming course names match

                        recommendation = Recommendation.objects.create(
                            user=request.user,
                            course=course,
                            similarity_score=rec['similarity_score'],
                            confidence_score=rec['confidence_score'],
                            combined_score=rec['combined_score']
                        )
                        created_recommendations.append(recommendation)
                    except Course.DoesNotExist:
                        print(f"Course not found: {rec['course']}")
                        continue
                    except Exception as e:
                        print(f"Error creating recommendation for {rec['course']}: {str(e)}")
                        continue

                if created_recommendations:
                    messages.success(request, "Recommendations generated successfully!")
                else:
                    messages.warning(request, "No valid recommendations could be generated.")

                return redirect('recommendations')

            except Exception as e:
                error_message = f"Error processing grades: {str(e)}"
                print(f"Error details: {error_message}")
                messages.error(request, error_message)

        return render(request, 'dashboard.html', {
            'selected_subjects': request.user.userprofile.selected_subjects.all(),
            'existing_grades': StudentSubject.objects.filter(user=request.user),
            'errors': serializer.errors
        })


class RecommendationView(APIView):
    @method_decorator(login_required)
    def get(self, request):
        recommendations = Recommendation.objects.filter(
            user=request.user
        )

        serializer = RecommendationSerializer(recommendations, many=True)
        return render(request, 'recommendations/recommendation.html', {
            'recommendations': serializer.data
        })
