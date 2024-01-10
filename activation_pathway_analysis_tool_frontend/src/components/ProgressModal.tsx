import React from 'react';
import '../styles/modal.css';

const Modal = ({ show, children }: {
    show: boolean,
    children: React.ReactNode
}) => {
  const showHideClassName = show ? 'progress-modal display-block' : 'progress-modal display-none';

  return (
    <div className={showHideClassName}>
      <section className='progress-modal-main'>
        {children}
      </section>
    </div>
  );
};

export default Modal;
