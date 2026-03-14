import streamlit as st
from frontend.pages.chat_page import render_chat_page
from frontend.pages.upload_page import render_upload_page
from frontend.pages.admin_page import render_admin_page
from frontend.utils.api_client import get_health


st.set_page_config(
    page_title='Enterprise AI Assistant',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)


# Sidebar navigation
with st.sidebar:
    st.title('🤖 AI Assistant')
    st.caption('Enterprise Knowledge Base')
    st.divider()

    page = st.radio(
        'Navigation',
        ['💬 Chat', '📁 Upload Documents', '📊 Admin Dashboard'],
        label_visibility='collapsed'
    )

    st.divider()

    # Backend status indicator
    health = get_health()
    if health:
        st.success('Backend: Online')
        st.caption(f'Vectors: {health["total_vectors"]}')
    else:
        st.error('Backend: Offline')
        st.caption('Start FastAPI server first')


# Render selected page
if page == '💬 Chat':
    render_chat_page()
elif page == '📁 Upload Documents':
    render_upload_page()
elif page == '📊 Admin Dashboard':
    render_admin_page()